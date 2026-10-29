"""Serve Lab web app. Runs fully on this computer.

uv run python -m app      # then open http://127.0.0.1:7860
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage data leaves the machine

import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from core import ROOT, history, load_config
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


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def phase_name(phase: str) -> str:
    return STROKE.LABELS.get(phase, "sin problema claro")  # phase "none": no_clear_problem


def feedback_markdown(feedback: dict, seconds: float, cached: bool = False) -> str:
    timing = f"Tiempo de análisis: {seconds:.0f} s" + (" (respuesta guardada)" if cached else "")
    drill = DRILLS[feedback["drill_id"]]
    lines = [
        "### No vemos un problema claro"
        if feedback["phase"] == "none"
        else f"### Fase a mejorar: {phase_name(feedback['phase'])}",
        feedback["observation"],
        f"**Ejercicio recomendado: {drill['name']}.** {drill['instructions']}",
        f"Confianza: {CONFIDENCE[feedback['confidence']]} · {timing}",
    ]
    if feedback["confidence"] == "low" or not feedback["consistent"]:
        lines.append(
            "_La evidencia es débil: toma esta observación con cautela y graba otro intento._"
        )
    return "\n\n".join(lines)


def comparison(previous: dict | None, current: dict):
    """Side-by-side key frame of the current main phase, and whether the main point changed."""
    if previous is None:
        return "Sube otro intento para compararlo con este.", None
    old, new = previous["feedback"], current["feedback"]
    if old["category"] == new["category"]:
        verdict = f"El punto a mejorar sigue siendo el mismo ({phase_name(new['phase'])})."
    else:
        verdict = (f"Cambió el punto principal: antes {phase_name(old['phase'])}, "
                   f"ahora {phase_name(new['phase'])}.")  # fmt: skip

    # key frame of the main phase now, or of the one flagged before when now there is none
    phase = next((p for p in (new["phase"], old["phase"]) if p in STROKE.LABELS), None)
    if phase is None:
        return f"**{verdict}**", None
    before, after = previous["images"][phase], current["images"][phase]
    height = min(before.shape[0], after.shape[0])
    pair = [
        cv2.resize(i, (round(i.shape[1] * height / i.shape[0]), height)) for i in (before, after)
    ]
    image = rgb(cv2.hconcat(pair))
    return f"**{verdict}** Izquierda: intento anterior; derecha: intento actual.", image


def run_analysis(video_path, progress) -> dict:
    """Everything that takes time. Returns {"retake": reason}, {"error": message} or the result."""
    start = time.perf_counter()
    try:
        progress(0.1, desc="Revisando el video…")
        frames = sample(video_path, **CONFIG["analysis"])
    except InsufficientEvidence as reason:
        return {"retake": str(reason)}
    except (ValueError, FileNotFoundError):
        return {"error": "No se pudo leer el video. Sube un archivo .mp4 o .mov de un solo saque."}

    try:
        progress(0.3, desc="Buscando las fases del saque…")
        frames, key_frames, _ = base.phase_frames(STROKE, frames, VLM_MODEL, VERSION, **GENERATION)
        images = {phase: frames.images[i] for phase, i in key_frames.items()}
        progress(0.6, desc="Generando la retroalimentación…")
        feedback = base.feedback(STROKE, images, VLM_MODEL, "full", VERSION, **GENERATION)
    except InsufficientEvidence as reason:
        return {"retake": str(reason)}
    except Exception:  # noqa: BLE001 - any model failure: ask to try again, never crash
        return {
            "error": "El modelo no respondió correctamente. Intenta analizar el video otra vez."
        }
    seconds = time.perf_counter() - start

    progress(0.85, desc="Preparando la cámara lenta…")
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


def analyze_attempt(video_path, session, progress=gr.Progress()):  # noqa: B008 - Gradio pattern
    empty = ("", None, None, "", None, session)
    if not video_path:
        return ("Sube un video de tu saque.", *empty)
    result = analysis(video_path, progress)
    if "retake" in result:
        return (f"### Necesitamos una nueva grabación\n{result['retake']}\n\n{RECORDING_TIPS}",
                *empty)  # fmt: skip
    if "error" in result:
        return (result["error"], *empty)

    feedback, images = result["feedback"], result["images"]
    evidence = set(feedback["evidence_phases"])
    gallery = [
        (rgb(images[p]), STROKE.LABELS[p] + (" ★" if p in evidence else "")) for p in STROKE.PHASES
    ]
    current = {"feedback": feedback, "images": images, "record": history.attempt(feedback)}
    compare_text, compare_image = comparison(session[-1] if session else None, current)
    return (
        f"Intento {len(session) + 1} analizado.",
        feedback_markdown(feedback, result["seconds"], result.get("cached", False)),
        result["slow"],
        gallery,
        compare_text,
        compare_image,
        [*session, current],
    )


def finish_session(player: str, session):
    if not session:
        return "Analiza al menos un intento antes de terminar la sesión.", session
    attempts = [s["record"] for s in session]
    player = player.strip()
    try:
        sessions = (
            history.save_session(player, attempts)
            if player
            else [{"date": "hoy", "attempts": attempts}]
        )
        summary = history.summarize(
            VLM_MODEL, PROMPTS["summary"], STROKE.LABELS, sessions, **GENERATION
        )
    except ValueError as error:
        return str(error), session
    saved = (
        f"Historial guardado para **{player}**."
        if player
        else "Sin ID de jugador: no se guardó el historial."
    )
    text = (f"### Resumen de la sesión\n{summary['summary']}\n\n"
            f"**Enfoque para la próxima práctica ({STROKE.LABELS[summary['focus_phase']]}):** "
            f"{summary['next_focus']}\n\n{saved}")  # fmt: skip
    return text, []


with gr.Blocks(title="Serve Lab", analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
    gr.Markdown(
        "# Serve Lab\nSube un video de tu saque grabado de lado. El asistente divide el saque en "
        "seis fases, señala la que más puedes mejorar con los cuadros que lo muestran y te "
        "propone un ejercicio. No reemplaza a un entrenador."
    )
    session = gr.State([])
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Video del saque", sources=["upload"])
            if DEMO_VIDEOS:
                gr.Examples([[str(p)] for p in DEMO_VIDEOS], inputs=[video_in],
                            label="Videos de la demo: elige uno y pulsa Analizar saque")  # fmt: skip
            player = gr.Textbox(
                label="ID de jugador (opcional, guarda tu historial)", placeholder="p01"
            )
            analyze_button = gr.Button("Analizar saque", variant="primary")
            finish_button = gr.Button("Terminar sesión")
            gr.Markdown(RECORDING_TIPS)
        with gr.Column(scale=2):
            status = gr.Markdown()
            feedback_out = gr.Markdown()
            slow_out = gr.Video(label="Cámara lenta con las fases", interactive=False)
    gallery_out = gr.Gallery(label="Cuadros clave (★ = evidencia de la observación)", columns=6)
    comparison_text = gr.Markdown()
    comparison_image = gr.Image(label="Intento anterior y actual", interactive=False)
    summary_out = gr.Markdown()

    analyze_button.click(
        analyze_attempt,
        [video_in, session],
        [status, feedback_out, slow_out, gallery_out, comparison_text, comparison_image, session],
    )
    finish_button.click(finish_session, [player, session], [summary_out, session])


if __name__ == "__main__":
    for n, path in enumerate(DEMO_VIDEOS, 1):
        DEMO_FOLDERS.add(cache_folder(path))
        result = analysis(path, lambda *args, **kwargs: None)
        answer = (
            result.get("feedback", {}).get("category") or result.get("retake") or result["error"]
        )
        print(f"Demo {n}/{len(DEMO_VIDEOS)}: {path.name}: {answer}", flush=True)
    demo.queue(default_concurrency_limit=1)  # one model on one GPU: analyze one video at a time
    demo.launch(server_name="127.0.0.1", max_file_size="200mb")
