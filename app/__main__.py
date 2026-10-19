"""Serve Lab web app. Runs fully on this computer.

uv run python -m app      # then open http://127.0.0.1:7860
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage data leaves the machine

import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from core import history, load_config
from core.analysis import InsufficientEvidence, analyze, write_overlay
from core.llm import VLM
from strokes import base

CONFIG = load_config()
STROKE = base.get("serve")
CRITERIA = base.load(STROKE, "criteria.yaml")
DRILLS = {d["id"]: d for d in base.load(STROKE, "drills.yaml")["drills"]}
VERSION = CONFIG["llm"]["prompt_version"]
PROMPTS = base.load(STROKE, f"prompts/{VERSION}.yaml")
GENERATION = {k: CONFIG["llm"][k] for k in ("temperature", "top_p", "max_tokens", "seed")}
CONFIDENCE = {"high": "alta", "medium": "media", "low": "baja"}
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="servelab-"))

RECORDING_TIPS = """**Cómo grabar el saque**
- Teléfono en trípode a la altura de la cadera, a 5–6 m, sin zoom.
- De lado: la cámara mira de frente al pecho cuando estás de perfil a la red.
- Todo el cuerpo en el cuadro, con espacio arriba para el lanzamiento.
- Un solo saque por video, de 3 a 5 segundos, a 60 o 120 fps, con buena luz."""

print(f"Cargando {CONFIG['llm']['model']}…")
VLM_MODEL = VLM(CONFIG["llm"]["model"])


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def feedback_markdown(feedback: dict, seconds: float) -> str:
    drill = DRILLS[feedback["drill_id"]]
    lines = [
        f"### Fase a mejorar: {STROKE.LABELS[feedback['phase']]}",
        feedback["observation"],
        f"**Ejercicio recomendado: {drill['name']}.** {drill['instructions']}",
        f"Confianza: {CONFIDENCE[feedback['confidence']]} · Tiempo de análisis: {seconds:.0f} s",
    ]
    if feedback["confidence"] == "low" or not feedback.get("consistent", True):
        lines.append(
            "_La evidencia es débil: toma esta observación con cautela y graba otro intento._"
        )
    if feedback["source"] == "rules":
        lines.append(
            "_El modelo no respondió correctamente; esta recomendación sale de reglas fijas._"
        )
    return "\n\n".join(lines)


def measurement_rows(measurements: dict[str, float]) -> list[list]:
    checks = {r["metric"]: r for r in base.check_results(CRITERIA, measurements)}
    rows = []
    for name, value in measurements.items():
        check = checks.get(name)
        limits = " y ".join(f"{k} {check[k]}" for k in ("min", "max") if check and k in check)
        status = "" if check is None else ("fuera de rango" if check["deviation"] > 0 else "bien")
        rows.append([STROKE.MEASUREMENT_LABELS.get(name, name), value, limits, status])
    return rows


def comparison(previous: dict | None, current: dict):
    """Side-by-side key frame of the current main phase, and how the measurements changed."""
    if previous is None:
        return "Sube otro intento para compararlo con este.", None
    phase = current["feedback"]["phase"]
    before, after = previous["overlays"][phase], current["overlays"][phase]
    height = min(before.shape[0], after.shape[0])
    pair = [
        cv2.resize(i, (round(i.shape[1] * height / i.shape[0]), height)) for i in (before, after)
    ]
    image = rgb(cv2.hconcat(pair))

    old, new = previous["feedback"], current["feedback"]
    if old["category"] == new["category"]:
        verdict = f"El punto a mejorar sigue siendo el mismo ({STROKE.LABELS[phase]})."
    else:
        verdict = (f"Cambió el punto principal: antes {STROKE.LABELS[old['phase']]}, "
                   f"ahora {STROKE.LABELS[phase]}.")  # fmt: skip
    changes = [
        f"- {STROKE.MEASUREMENT_LABELS.get(k, k)}: {previous['measurements'][k]} → {v}"
        for k, v in current["measurements"].items()
        if k in previous["measurements"] and v != previous["measurements"][k]
    ]
    text = f"**{verdict}** Izquierda: intento anterior; derecha: intento actual.\n\n" + "\n".join(
        changes
    )
    return text, image


def analyze_attempt(video_path, session, progress=gr.Progress()):  # noqa: B008 - Gradio pattern
    empty = ("", None, None, [], "", None, session)
    if not video_path:
        return ("Sube un video de tu saque.", *empty)
    pose = CONFIG["pose"]
    try:
        progress(0.1, desc="Detectando la pose…")
        analysis = analyze(video_path, STROKE, pose["backend"], pose[pose["backend"]],
                           cache=False, **CONFIG["analysis"])  # fmt: skip
    except InsufficientEvidence as reason:
        return (f"### Necesitamos una nueva grabación\n{reason}\n\n{RECORDING_TIPS}", *empty)
    except (ValueError, FileNotFoundError):
        return ("No se pudo leer el video. Sube un archivo .mp4 o .mov de un solo saque.", *empty)

    progress(0.5, desc="Generando la retroalimentación…")
    try:
        feedback = base.feedback(STROKE, analysis, VLM_MODEL, "full", VERSION, **GENERATION)
    except Exception:  # noqa: BLE001 - any model failure falls back to the rule baseline
        feedback = base.rule_feedback(STROKE, analysis.measurements)
    seconds = analysis.seconds + feedback.get("llm_seconds", 0)

    progress(0.85, desc="Preparando la cámara lenta…")
    overlay = write_overlay(analysis, OUTPUT_DIR / f"overlay-{len(session) + 1}-{os.getpid()}.mp4",
                            STROKE.LABELS, CONFIG["app"]["slowdown"])  # fmt: skip
    evidence = set(feedback["evidence_phases"])
    gallery = [
        (rgb(analysis.overlays[p]), STROKE.LABELS[p] + (" ★" if p in evidence else ""))
        for p in STROKE.PHASES
    ]
    current = {
        "feedback": feedback,
        "measurements": analysis.measurements,
        "overlays": analysis.overlays,
        "record": history.attempt(feedback, analysis.measurements),
    }
    compare_text, compare_image = comparison(session[-1] if session else None, current)
    return (
        f"Intento {len(session) + 1} analizado.",
        feedback_markdown(feedback, seconds),
        str(overlay),
        gallery,
        measurement_rows(analysis.measurements),
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
        "propone un ejercicio. No reemplaza a tu entrenador."
    )
    session = gr.State([])
    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Video del saque", sources=["upload"])
            player = gr.Textbox(
                label="ID de jugador (opcional, guarda tu historial)", placeholder="p01"
            )
            analyze_button = gr.Button("Analizar saque", variant="primary")
            finish_button = gr.Button("Terminar sesión")
            gr.Markdown(RECORDING_TIPS)
        with gr.Column(scale=2):
            status = gr.Markdown()
            feedback_out = gr.Markdown()
            overlay_out = gr.Video(label="Cámara lenta con la pose", interactive=False)
    gallery_out = gr.Gallery(label="Cuadros clave (★ = evidencia de la observación)", columns=6)
    comparison_text = gr.Markdown()
    comparison_image = gr.Image(label="Intento anterior y actual", interactive=False)
    with gr.Accordion("Mediciones", open=False):
        table = gr.Dataframe(headers=["Medición", "Valor", "Rango del entrenador", "Estado"],
                             interactive=False)  # fmt: skip
    summary_out = gr.Markdown()

    analyze_button.click(
        analyze_attempt,
        [video_in, session],
        [
            status,
            feedback_out,
            overlay_out,
            gallery_out,
            table,
            comparison_text,
            comparison_image,
            session,
        ],
    )
    finish_button.click(finish_session, [player, session], [summary_out, session])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)  # one model on one GPU: analyze one video at a time
    demo.launch(server_name="127.0.0.1", max_file_size="200mb")
