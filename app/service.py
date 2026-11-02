"""What the Serve Lab apps share: the model, one analysis of a video, and the demo cache.

Used by the Gradio app (app/__main__.py) and by the web app's API (app/api.py).
"""

import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2

from core import ROOT, load_config
from core.analysis import InsufficientEvidence, sample, write_slow_motion
from core.llm import VLM
from strokes import base

CONFIG = load_config()
STROKE = base.get("serve")
DRILLS = {d["id"]: d for d in base.load(STROKE, "drills.yaml")["drills"]}
VERSION = CONFIG["llm"]["prompt_version"]
PROMPTS = base.load(STROKE, f"prompts/{VERSION}.yaml")
GENERATION = {k: CONFIG["llm"][k] for k in ("temperature", "top_p", "max_tokens", "seed")}
CONFIDENCE = {"high": "alta", "medium": "media", "low": "baja"}
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="servelab-"))
CRITERIA_VERSION = base.load(STROKE, "criteria.yaml")["version"]
CATEGORIES = base.load(STROKE, "criteria.yaml")["categories"]

# Demo mode (uv run python -m app --demo): the videos in data/demo/ are analyzed once at start
# and their answers saved in data/processed/demo_cache/, so at the fair they show at once.
DEMO = "--demo" in sys.argv
DEMO_DIR = ROOT / "data" / "demo"
CACHE_DIR = ROOT / "data" / "processed" / "demo_cache"
DEMO_VIDEOS = sorted(p for p in DEMO_DIR.glob("*") if DEMO and p.suffix.lower() in (".mp4", ".mov"))
DEMO_FOLDERS: set[Path] = set()  # cache folders of the demo videos; visitors' videos are not saved

RECORDING_TIPS = """**Cómo grabar el saque** (guía completa: docs/grabacion.md)
- Teléfono horizontal, a la altura del pecho y recto, a unos 6 m, lente 1x sin zoom.
- De lado: la cámara mira tu pecho, sobre la prolongación de la línea de fondo.
- Todo el cuerpo y la raqueta arriba dentro del cuadro, sin el sol detrás de ti.
- Un solo saque por video, de 4 a 8 segundos, a 1080p y 60 fps."""

print(f"Cargando {CONFIG['llm']['model']}…")
VLM_MODEL = VLM(CONFIG["llm"]["model"])


def phase_name(phase: str) -> str:
    return STROKE.LABELS.get(phase, "sin problema claro")  # phase "none": no_clear_problem


def certainty_text(checks: dict) -> str:
    """How far the answer can be trusted: how many checks the model could judge, and which
    phases it could not see."""
    answers = [c["answer"] for c in checks.values()]
    unclear = [p for p in STROKE.PHASES
               if any(c["answer"] == "unclear" and judged_on(k, c) == p
                      for k, c in checks.items())]  # fmt: skip
    text = f"Pudimos revisar {len(answers) - answers.count('unclear')} de {len(answers)} puntos."
    if unclear:
        text += f" No se ve con claridad: {', '.join(phase_name(p) for p in unclear)}."
    if answers.count("unclear") * 2 >= len(answers):
        text += (" Con tantos puntos sin ver, la respuesta dice poco: tómala con cautela y graba "
                 "otro intento siguiendo la guía de grabación.")  # fmt: skip
    return text


def certainty(feedback: dict) -> str:
    """certainty_text, or a warning for versions without checks (before v6)."""
    if feedback.get("checks"):
        return certainty_text(feedback["checks"])
    if feedback["confidence"] == "low" or not feedback["consistent"]:
        return "La evidencia es débil: toma esta observación con cautela y graba otro intento."
    return ""


def check_text(category: str, check: dict) -> tuple[str, str]:
    """Body part and the state the model saw, in Spanish."""
    part, good, bad = CATEGORIES[category]["show"]
    return part, {"yes": good, "no": bad}.get(check["answer"], "no se ve claro")


def judged_on(category: str, check: dict) -> str:
    """The phase whose key frame the check looked at (v10 can judge on another phase)."""
    return check.get("phase") or CATEGORIES[category]["phase"]


def phase_verdict(checks: dict, phase: str) -> str | None:
    """The verdict of one phase: "no" if a check found its problem, "yes" if all looked
    right, "unclear" otherwise; None when the phase has no checks (the stance)."""
    answers = [c["answer"] for k, c in checks.items() if judged_on(k, c) == phase]
    if not answers:
        return None
    return "no" if "no" in answers else "yes" if set(answers) == {"yes"} else "unclear"


def run_analysis(video_path, progress) -> dict:
    """Everything that takes time. Returns {"retake": reason}, {"error": message} or the result."""
    start = time.perf_counter()
    try:
        progress(0.1, desc="Paso 1 de 4: revisando el video… (unos 30 s en total)")
        frames = sample(video_path, **CONFIG["analysis"])
    except InsufficientEvidence as reason:
        return {"retake": str(reason)}
    except (ValueError, FileNotFoundError):
        return {"error": "No se pudo leer el video. Sube un archivo .mp4 o .mov de un solo saque."}

    try:
        progress(0.3, desc="Paso 2 de 4: buscando las fases del saque…")
        frames, key_frames, _ = base.phase_frames(STROKE, frames, VLM_MODEL, VERSION, **GENERATION)
        images = {phase: frames.images[i] for phase, i in key_frames.items()}
        progress(0.6, desc="Paso 3 de 4: revisando cada fase…")
        feedback = base.feedback(STROKE, images, VLM_MODEL, "full", VERSION, **GENERATION)
    except InsufficientEvidence as reason:
        return {"retake": str(reason)}
    except Exception:  # noqa: BLE001 - any model failure: ask to try again, never crash
        return {
            "error": "El modelo no respondió correctamente. Intenta analizar el video otra vez."
        }
    seconds = time.perf_counter() - start

    progress(0.85, desc="Paso 4 de 4: preparando la cámara lenta…")
    slow = write_slow_motion(frames, key_frames, STROKE.LABELS,
                             OUTPUT_DIR / f"slow-{time.time_ns()}.mp4",
                             CONFIG["app"]["slowdown"])  # fmt: skip
    return {"feedback": feedback, "images": images, "slow": str(slow), "seconds": seconds}


def cache_folder(video_path) -> Path:
    """One folder per video content, model, prompt version and criteria version."""
    digest = hashlib.sha256(Path(video_path).read_bytes())
    digest.update(f"{CONFIG['llm']['model']}|{VERSION}|{CRITERIA_VERSION}".encode())
    return CACHE_DIR / digest.hexdigest()[:16]


def analysis(video_path, progress) -> dict:
    """run_analysis; for a demo video the answer is saved once and then shown at once."""
    folder = cache_folder(video_path) if DEMO_FOLDERS else None
    if folder not in DEMO_FOLDERS:
        return run_analysis(video_path, progress)
    if (folder / "result.json").exists():
        result = json.loads((folder / "result.json").read_text())
        if "feedback" in result:
            result["images"] = {p: cv2.imread(str(folder / f"{p}.jpg")) for p in STROKE.PHASES}
            result["slow"] = str(folder / "slow.mp4")
        return {**result, "cached": True}

    result = run_analysis(video_path, progress)
    if "error" in result:  # model failures are not saved: the next try may work
        return result
    folder.mkdir(parents=True, exist_ok=True)
    if "feedback" in result:
        for phase, image in result["images"].items():
            cv2.imwrite(str(folder / f"{phase}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy(result["slow"], folder / "slow.mp4")
    saved = {k: v for k, v in result.items() if k not in ("images", "slow")}
    (folder / "result.json").write_text(json.dumps(saved, ensure_ascii=False))  # written last
    return result


def prepare_demo():
    """Analyze each demo video once (or load its saved answer) before the app opens."""
    for n, path in enumerate(DEMO_VIDEOS, 1):
        DEMO_FOLDERS.add(cache_folder(path))
        result = analysis(path, lambda *args, **kwargs: None)
        answer = (
            result.get("feedback", {}).get("category") or result.get("retake") or result["error"]
        )
        print(f"Demo {n}/{len(DEMO_VIDEOS)}: {path.name}: {answer}", flush=True)
