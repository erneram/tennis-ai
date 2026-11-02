"""Serve Lab web app: the API the React frontend in web/ calls, and the built frontend.

uv run python -m app.api [--demo]     # http://127.0.0.1:7860 (build web/ first: npm run build)

An analysis runs in a background thread; the page polls its job for progress and the result.
"""

import re
import shutil
import threading
import uuid
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile

from app.service import (
    CATEGORIES,
    CONFIDENCE,
    CONFIG,
    DEMO_VIDEOS,
    DRILLS,
    GENERATION,
    OUTPUT_DIR,
    PROMPTS,
    RECORDING_TIPS,
    STROKE,
    VERSION,
    VLM_MODEL,
    analysis,
    certainty,
    check_text,
    judged_on,
    phase_name,
    phase_verdict,
    prepare_demo,
)
from core import ROOT, history

MAX_BYTES = 200 * 1024 * 1024
FILES = {f"{p}.jpg" for p in STROKE.PHASES} | {"slow.mp4"}  # what a job folder may serve
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="Serve Lab", docs_url=None, redoc_url=None, openapi_url=None)
JOBS: dict[str, dict] = {}
MODEL_LOCK = threading.Lock()  # one model on one GPU: one call at a time


@app.get("/api/config")
def config():
    model = re.sub(r"-Instruct.*", "", CONFIG["llm"]["model"].split("/")[-1])
    demo = [{"id": p.name, "kind": "pro" if p.stem.startswith("pro") else "amateur",
             "label": ("Profesional" if p.stem.startswith("pro") else "Amateur") + f" · {p.stem}",
             "video_url": f"/api/demo/{p.name}"} for p in DEMO_VIDEOS]  # fmt: skip
    tips = [line[2:] for line in RECORDING_TIPS.splitlines() if line.startswith("- ")]
    phases = [{"id": p, "label": STROKE.LABELS[p]} for p in STROKE.PHASES]
    return {"model": model, "version": VERSION, "phases": phases, "demo": demo, "tips": tips}


@app.get("/api/demo/{name}")
def demo_video(name: str):
    path = next((p for p in DEMO_VIDEOS if p.name == name), None)
    if path is None:
        raise HTTPException(404)
    return FileResponse(path)


@app.post("/api/analyze")
async def analyze(request: Request):
    """Multipart `file` (a visitor's video) or JSON {"demo": name}. Returns the job to poll."""
    job = uuid.uuid4().hex[:12]
    folder = OUTPUT_DIR / job
    if request.headers.get("content-type", "").startswith("multipart/"):
        upload = (await request.form()).get("file")
        if not isinstance(upload, UploadFile):
            raise HTTPException(400, "Falta el video.")
        if (upload.size or 0) > MAX_BYTES:
            raise HTTPException(413, "El video pesa más de 200 MB: graba un solo saque.")
        suffix = Path(upload.filename or "").suffix.lower()
        folder.mkdir()
        path = folder / f"video{suffix if suffix in ('.mp4', '.mov', '.m4v') else '.mp4'}"
        with path.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
    else:
        try:
            body = await request.json()
        except ValueError:
            body = None
        name = body.get("demo") if isinstance(body, dict) else None
        path = next((p for p in DEMO_VIDEOS if p.name == name), None)
        if path is None:
            raise HTTPException(404, "Ese video de la demo no existe.")
        folder.mkdir()
    JOBS[job] = {"status": "queued", "step": 0, "progress": 0.0, "message": "En espera…"}
    threading.Thread(target=run_job, args=(job, path), daemon=True).start()
    return {"job": job}


def run_job(job: str, path: Path):
    state = JOBS[job]

    def progress(value, desc=""):  # same calls as gr.Progress: "Paso n de 4: …"
        step = int(desc[5]) if desc.startswith("Paso ") else state["step"]
        state.update(step=step, progress=value, message=desc)

    with MODEL_LOCK:
        state["status"] = "running"
        try:
            result = to_json(job, analysis(path, progress))
        except Exception:  # noqa: BLE001 - the page must always get an answer
            result = {"kind": "error", "title": "No pudimos analizar el video",
                      "message": "Algo falló al analizar el video. Intenta otra vez."}  # fmt: skip
    state.update(status="done", step=4, progress=1.0, message="", result=result)


@app.get("/api/jobs/{job}")
def job_state(job: str):
    if job not in JOBS:
        raise HTTPException(404)
    return dict(JOBS[job])


@app.get("/api/files/{job}/{name}")
def job_file(job: str, name: str):
    if job not in JOBS or name not in FILES:
        raise HTTPException(404)
    return FileResponse(OUTPUT_DIR / job / name)


def to_json(job: str, result: dict) -> dict:
    """The answer for the page; the key frames and slow motion are saved for /api/files."""
    if "retake" in result:
        return {"kind": "retake", "title": "Necesitamos una nueva grabación",
                "message": result["retake"]}  # fmt: skip
    if "error" in result:
        return {"kind": "error", "title": "No pudimos analizar el video",
                "message": result["error"]}  # fmt: skip
    folder = OUTPUT_DIR / job
    for phase, image in result["images"].items():
        cv2.imwrite(str(folder / f"{phase}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    shutil.copy(result["slow"], folder / "slow.mp4")

    feedback = result["feedback"]
    checks = feedback.get("checks", {})
    phase = None if feedback["phase"] == "none" else feedback["phase"]
    seen_in = next((p for p in feedback["evidence_phases"] if p != feedback["phase"]), None)
    drill = DRILLS[feedback["drill_id"]]
    answers = [c["answer"] for c in checks.values()]
    return {
        "kind": "feedback",
        "seconds": round(result["seconds"], 1),
        "cached": result.get("cached", False),
        "feedback": {
            "phase": phase,
            "phase_label": phase and phase_name(phase),
            "category": feedback["category"],
            "observation": feedback["observation"],
            "seen_in": seen_in,
            "seen_in_label": seen_in and phase_name(seen_in),
            "drill": {"name": drill["name"], "instructions": drill["instructions"]},
            "confidence": feedback["confidence"],
            "confidence_label": CONFIDENCE[feedback["confidence"]],
            "certainty": certainty(feedback),
            "checked": len(answers) - answers.count("unclear"),
            "total": len(answers),
        },
        "phases": [
            {
                "id": p,
                "label": STROKE.LABELS[p],
                "verdict": phase_verdict(checks, p),
                "evidence": p in feedback["evidence_phases"],
                "image_url": f"/api/files/{job}/{p}.jpg",
            }
            for p in STROKE.PHASES
        ],
        "checks": [
            {
                "category": c,
                "phase": judged_on(c, check),
                "phase_label": phase_name(judged_on(c, check)),
                "part": check_text(c, check)[0],
                "state": check_text(c, check)[1],
                "answer": check["answer"],
                "seen": check.get("seen", ""),
            }
            for c, check in checks.items()
            if c in CATEGORIES
        ],
        "slow_url": f"/api/files/{job}/slow.mp4",
        "record": history.attempt(feedback),
    }


def valid_attempt(attempt) -> bool:
    """Attempts come back from the page and go into the history and the summary prompt."""
    return (isinstance(attempt, dict) and set(attempt) == {"time", "phase", "category", "drill_id"}
            and re.fullmatch(r"\d\d:\d\d", str(attempt["time"])) is not None
            and attempt["phase"] in (*STROKE.PHASES, "none")
            and attempt["category"] in (*CATEGORIES, "no_clear_problem")
            and attempt["drill_id"] in DRILLS)  # fmt: skip


@app.post("/api/summary")
def summary(body: dict):
    attempts, player = body.get("attempts") or [], str(body.get("player") or "").strip()
    if not attempts:
        return {"error": "Analiza al menos un intento antes de terminar la sesión."}
    if not isinstance(attempts, list) or not all(valid_attempt(a) for a in attempts):
        return {"error": "Los intentos de la sesión no son válidos. Empieza de nuevo."}
    try:
        with MODEL_LOCK:
            sessions = (history.save_session(player, attempts) if player
                        else [{"date": "hoy", "attempts": attempts}])  # fmt: skip
            answer = history.summarize(
                VLM_MODEL, PROMPTS["summary"], STROKE.LABELS, sessions, **GENERATION
            )
    except ValueError as error:  # an invalid player ID
        return {"error": str(error)}
    except Exception:  # noqa: BLE001 - model failure
        return {"error": "El modelo no respondió correctamente. Intenta otra vez."}
    return {"text": answer["summary"], "focus_label": STROKE.LABELS[answer["focus_phase"]],
            "next_focus": answer["next_focus"], "saved": bool(player)}  # fmt: skip


if WEB_DIST.exists():  # after the API routes: everything else is the page
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


if __name__ == "__main__":
    prepare_demo()
    if not WEB_DIST.exists():
        print("Falta web/dist: cd web && npm install && npm run build", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=7860)
