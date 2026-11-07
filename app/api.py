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
    CONFIDENCE,
    DEMO_VIDEOS,
    DRILLS,
    MODEL,
    MOMENTS,
    OUTPUT_DIR,
    PARTS,
    RECORDING_TIPS,
    STROKE,
    analysis,
    certainty,
    confidence,
    observation,
    prepare_demo,
    reason,
)
from core import ROOT, history
from strokes.serve import baseline, model

MAX_BYTES = 200 * 1024 * 1024
FILES = {f"{m}.jpg" for m in MOMENTS} | {"slow.mp4"}  # what a job folder may serve
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="Serve Lab", docs_url=None, redoc_url=None, openapi_url=None)
JOBS: dict[str, dict] = {}
MODEL_LOCK = threading.Lock()  # one analysis at a time: MediaPipe uses every core


@app.get("/api/config")
def config():
    demo = [{"id": p.name, "kind": "pro" if p.stem.startswith("pro") else "amateur",
             "label": ("Profesional" if p.stem.startswith("pro") else "Amateur") + f" · {p.stem}",
             "video_url": f"/api/demo/{p.name}"} for p in DEMO_VIDEOS]  # fmt: skip
    tips = [line[2:] for line in RECORDING_TIPS.splitlines() if line.startswith("- ")]
    return {"model": f"MediaPipe + {MODEL['name'].upper()}", "version": "modelo propio",
            "demo": demo, "tips": tips}  # fmt: skip


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

    def progress(value, desc=""):  # "Paso n de 4: …"
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
    """The answer for the page; the key moments and slow motion are saved for /api/files."""
    if "retake" in result:
        return {"kind": "retake", "title": "Necesitamos una nueva grabación",
                "message": result["retake"]}  # fmt: skip
    if "error" in result:
        return {"kind": "error", "title": "No pudimos analizar el video",
                "message": result["error"]}  # fmt: skip
    folder = OUTPUT_DIR / job
    for name, image in result["images"].items():
        cv2.imwrite(str(folder / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    shutil.copy(result["slow"], folder / "slow.mp4")

    answer = result["answer"]
    drops = answer["drops_early"]
    category = baseline.POSITIVE if drops else baseline.NEGATIVE
    drill_id = "toss_hold" if drops else "steady_serves"
    drill = DRILLS[drill_id]
    level = confidence(answer)
    return {
        "kind": "feedback",
        "seconds": round(result["seconds"], 1),
        "cached": result.get("cached", False),
        "feedback": {
            "drops_early": drops,
            "title": (
                "El brazo que lanza baja antes de tiempo"
                if drops
                else "No vemos que el brazo que lanza baje antes de tiempo"
            ),
            "phase_label": STROKE.LABELS["trophy"],
            "observation": observation(answer),
            "reason": reason(answer),
            "drill": {"name": drill["name"], "instructions": drill["instructions"]},
            "probability": round(answer["probability"], 3),
            "threshold": round(answer["threshold"], 3),
            "seeds": answer["seeds"],
            "confidence": level,
            "confidence_label": CONFIDENCE[level],
            "certainty": certainty(answer),
        },
        "moments": [
            {
                "id": name,
                "label": MOMENTS[name],
                "evidence": drops and name == "arm_down",
                "image_url": f"/api/files/{job}/{name}.jpg",
            }
            for name in MOMENTS
            if name in result["images"]
        ],
        "occlusion": {
            "parts": PARTS,
            "starts": [round(model.seconds(s), 2) for s in model.STARTS],
            "values": answer["occlusion"],
        },
        "slow_url": f"/api/files/{job}/slow.mp4",
        "record": history.attempt(category, answer["probability"], drill_id),
    }


def valid_attempt(attempt) -> bool:
    """Attempts come back from the page and go into the history and the summary."""
    return (isinstance(attempt, dict) and set(attempt) == {"time", "category", "probability",
                                                           "drill_id"}
            and re.fullmatch(r"\d\d:\d\d", str(attempt["time"])) is not None
            and attempt["category"] in (baseline.POSITIVE, baseline.NEGATIVE)
            and isinstance(attempt["probability"], (int, float))
            and 0 <= attempt["probability"] <= 1
            and attempt["drill_id"] in DRILLS)  # fmt: skip


def session_summary(attempts: list[dict]) -> dict:
    """What the session showed, from the attempts alone: how often the arm dropped early and how
    the probability moved, and the drill of the last attempt."""
    drops = sum(a["category"] == baseline.POSITIVE for a in attempts)
    text = f"En {len(attempts)} intento(s), el brazo que lanza bajó antes de tiempo en {drops}."
    if len(attempts) > 1:
        first, last = attempts[0]["probability"], attempts[-1]["probability"]
        trend = "bajó" if last < first else "subió" if last > first else "no cambió"
        text += (f" La probabilidad de que baje antes {trend}: del {first:.0%} en el primero al "
                 f"{last:.0%} en el último.")  # fmt: skip
    drill = DRILLS[attempts[-1]["drill_id"]]
    return {"text": text, "focus_label": drill["name"], "next_focus": drill["instructions"]}


@app.post("/api/summary")
def summary(body: dict):
    attempts, player = body.get("attempts") or [], str(body.get("player") or "").strip()
    if not attempts:
        return {"error": "Analiza al menos un intento antes de terminar la sesión."}
    if not isinstance(attempts, list) or not all(valid_attempt(a) for a in attempts):
        return {"error": "Los intentos de la sesión no son válidos. Empieza de nuevo."}
    if player:
        try:
            history.save_session(player, attempts)
        except ValueError as error:  # an invalid player ID
            return {"error": str(error)}
    return {**session_summary(attempts), "saved": bool(player)}


if WEB_DIST.exists():  # after the API routes: everything else is the page
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


if __name__ == "__main__":
    prepare_demo()
    if not WEB_DIST.exists():
        print("Falta web/dist: cd web && npm install && npm run build", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=7860)
