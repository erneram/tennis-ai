"""Serve Lab web app. Runs fully on this computer.

uv run python -m app      # then open http://127.0.0.1:7860
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage data leaves the machine

import hashlib
import html
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
CATEGORIES = base.load(STROKE, "criteria.yaml")["categories"]
# Answer of a check (v6 on): icon in the checklist and border color (BGR) of the key frame
MARKS = {
    "yes": ("✅", (80, 175, 76)),
    "no": ("❌", (54, 67, 244)),
    "unclear": ("❔", (158, 158, 158)),
}

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

# Look of the poster: its blue, ink and paper. System fonts only: the app must work offline.
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.stone,
    font=[
        "-apple-system",
        "BlinkMacSystemFont",
        "Segoe UI",
        "Helvetica Neue",
        "Arial",
        "sans-serif",
    ],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
).set(
    body_background_fill="#F6F6F1",
    body_text_color="#14202B",
    button_primary_background_fill="#1F5FA8",
    button_primary_background_fill_hover="#14407A",
    button_primary_text_color="#FFFFFF",
)
# Colors are set on every custom block, so they stay readable when the browser is in dark mode
CSS = """
.sl-header {background: #1F5FA8; color: #FFFFFF; border-radius: 16px; padding: 24px 28px;
  display: flex; flex-wrap: wrap; gap: 12px 24px; justify-content: space-between; align-items: flex-end}
.sl-header h1 {margin: 0; color: #FFFFFF; font-size: 42px; font-weight: 800; letter-spacing: -0.02em}
.sl-header p {margin: 6px 0 0; color: #FFFFFF; font-size: 17px; line-height: 1.45; max-width: 46em}
.sl-header .sl-kicker {color: #D7E6F7; font-size: 13px; font-weight: 700; letter-spacing: 0.08em}
.sl-badges {display: flex; flex-wrap: wrap; gap: 8px}
.sl-badge {display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 14px; font-weight: 600}
.sl-local {background: #FFFFFF; color: #1F5FA8}
.sl-card {background: #FFFFFF; color: #14202B; border: 1px solid #D9DCD3; border-left: 8px solid #9E9E9E;
  border-radius: 16px; padding: 18px 22px}
.sl-card.sl-no {border-left-color: #D93C3C}
.sl-card.sl-ok {border-left-color: #3C9A4F}
.sl-card.sl-warn {border-left-color: #E0A100}
.sl-card ol {margin: 8px 0 0; padding-left: 22px; line-height: 1.6}
.sl-card .sl-kicker {color: #5B6570; font-size: 13px; font-weight: 600}
.sl-card h2 {margin: 4px 0 2px; color: #14202B; font-size: 26px; line-height: 1.2}
.sl-card p {color: #14202B; margin: 8px 0}
.sl-card .sl-where {margin: 0; color: #3D4752}
.sl-card .sl-obs {font-size: 17px; line-height: 1.5}
.sl-drill {background: #F1F6D0; color: #14202B; border-radius: 12px; padding: 12px 16px; margin: 12px 0;
  line-height: 1.45}
.sl-meta {display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; color: #3D4752; font-size: 14px}
.sl-conf-high {background: #DDF0E0; color: #1E6B30}
.sl-conf-medium {background: #FCEBC7; color: #7A5200}
.sl-conf-low {background: #E5E7EB; color: #374151}
.sl-strip {display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px; margin-top: 14px}
.sl-chip {background: #F3F4F1; color: #14202B; border: 2px solid #E2E4DE; border-radius: 10px;
  padding: 8px 6px; text-align: center; font-size: 13px; line-height: 1.25}
.sl-chip b {display: block; font-size: 18px; color: #14202B !important}
.sl-chip.yes {background: #E6F4E8; border-color: #4CAF50}
.sl-chip.no {background: #FDE8E8; border-color: #F44336}
.sl-chip.unclear {background: #EEEEEE; border-color: #9E9E9E}
.sl-chip.unclear b, .sl-chip.none b {color: #5B6570 !important}
/* the theme colors bold text and spans in dark mode: keep these readable on the light card */
.sl-drill b, .sl-card li, .sl-card li b {color: #14202B !important}
.sl-meta .sl-cert {color: #3D4752 !important}
.sl-legend {color: #5B6570; font-size: 13px; margin-top: 6px}
@media (max-width: 640px) {.sl-strip {grid-template-columns: repeat(3, 1fr)} .sl-header h1 {font-size: 34px}}
"""
HEADER = """<div class="sl-header"><div>
<div class="sl-kicker">CC3092 DEEP LEARNING · UVG · PROYECTO 3</div>
<h1>Serve Lab</h1>
<p>Sube un video de tu saque grabado de lado. El asistente divide el saque en seis fases, señala
la que más puedes mejorar con los cuadros que lo muestran y te propone un ejercicio.</p></div>
<div class="sl-badges"><span class="sl-badge sl-local">100 % local · sin internet</span>
<span class="sl-badge sl-local">No reemplaza a un entrenador</span></div></div>"""

WELCOME = """<div class="sl-card"><h2>Cómo funciona</h2><ol>
<li>Elige un video de la demo o sube el tuyo, grabado de lado.</li>
<li>Pulsa <b>Analizar saque</b>: tarda unos 30 s, todo en esta computadora.</li>
<li>Mira la fase a mejorar, el ejercicio y qué tan seguro está el asistente en cada fase.</li>
</ol></div>"""


def notice_html(title: str, text: str) -> str:
    """A request to record again or to retry, as a card."""
    return (f'<div class="sl-card sl-warn"><h2>{html.escape(title)}</h2>'
            f"<p>{html.escape(text)}</p></div>")  # fmt: skip


print(f"Cargando {CONFIG['llm']['model']}…")
VLM_MODEL = VLM(CONFIG["llm"]["model"])


def rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def phase_name(phase: str) -> str:
    return STROKE.LABELS.get(phase, "sin problema claro")  # phase "none": no_clear_problem


def result_html(feedback: dict, seconds: float, attempt: int, cached: bool = False) -> str:
    """The answer as one card: the phase to improve, why, the drill, how far to trust it, and
    the verdict of every phase. Model text is escaped."""
    e = html.escape
    drill = DRILLS[feedback["drill_id"]]
    checks = feedback.get("checks", {})
    if feedback["phase"] == "none":
        tone, title = "sl-ok", "No vemos un problema claro"
    else:
        tone, title = "sl-no", f"Fase a mejorar: {phase_name(feedback['phase'])}"
    # a check judged on another phase's frame (v10 on): say where the ❌ is
    seen_in = [p for p in feedback["evidence_phases"] if p != feedback["phase"]]
    where = f'<p class="sl-where">Se ve en {e(phase_name(seen_in[0]))}</p>' if seen_in else ""
    if checks:
        certainty = certainty_text(checks)
    elif feedback["confidence"] == "low" or not feedback["consistent"]:
        certainty = "La evidencia es débil: toma esta observación con cautela y graba otro intento."
    else:
        certainty = ""
    timing = f"{seconds:.0f} s" + (" · respuesta guardada" if cached else "")
    confidence = feedback["confidence"]
    strip = phase_strip(checks, set(feedback["evidence_phases"])) if checks else ""
    return f"""<div class="sl-card {tone}">
<div class="sl-kicker">Intento {attempt} · análisis en {timing}</div>
<h2>{e(title)}</h2>{where}
<p class="sl-obs">{e(feedback["observation"])}</p>
<div class="sl-drill"><b>Ejercicio: {e(drill["name"])}.</b> {e(drill["instructions"])}</div>
<div class="sl-meta"><span class="sl-badge sl-conf-{confidence}">Confianza
{CONFIDENCE[confidence]}</span><span class="sl-cert">{e(certainty)}</span></div>{strip}</div>"""


def phase_strip(checks: dict, evidence: set[str]) -> str:
    """The six phases in order, each with the verdict of its checks (★: shown in the
    observation)."""
    chips = []
    for phase in STROKE.PHASES:
        verdict = phase_verdict(checks, phase)
        mark = {"yes": "✅", "no": "❌", "unclear": "?"}.get(verdict, "—")  # ❔ is faint
        star = " ★" if phase in evidence else ""
        chips.append(f'<div class="sl-chip {verdict or "none"}"><b>{mark}{star}</b>'
                     f"{html.escape(STROKE.LABELS[phase])}</div>")  # fmt: skip
    return (f'<div class="sl-strip">{"".join(chips)}</div><div class="sl-legend">✅ se ve bien'
            " · ❌ a mejorar · ? no se ve claro · — sin revisar · ★ lo muestra la observación</div>")  # fmt: skip


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


def judged_on(category: str, check: dict) -> str:
    """The phase whose key frame the check looked at (v10 can judge on another phase)."""
    return check.get("phase") or CATEGORIES[category]["phase"]


def checklist_markdown(checks: dict) -> str:
    """One line per check: phase, body part and the state the model saw."""
    lines = []
    for category, check in checks.items():
        part, good, bad = CATEGORIES[category]["show"]
        state = {"yes": good, "no": bad}.get(check["answer"], "no se ve claro")
        phase = phase_name(judged_on(category, check))
        lines.append(f"- {MARKS[check['answer']][0]} **{phase}** · {part}: {state}")
    return "\n".join(lines)


def phase_verdict(checks: dict, phase: str) -> str | None:
    """The verdict of one phase: "no" if a check found its problem, "yes" if all looked
    right, "unclear" otherwise; None when the phase has no checks (the stance)."""
    answers = [c["answer"] for k, c in checks.items() if judged_on(k, c) == phase]
    if not answers:
        return None
    return "no" if "no" in answers else "yes" if set(answers) == {"yes"} else "unclear"


def framed(image: np.ndarray, verdict: str | None) -> np.ndarray:
    """Key frame with a border colored by the verdict of its phase."""
    if verdict is None:
        return image
    return cv2.copyMakeBorder(image, *[14] * 4, cv2.BORDER_CONSTANT, value=MARKS[verdict][1])


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

    # frame that shows the main point now, or the one flagged before when now there is none
    # (the evidence frame: v10 judges some categories on another phase's key frame)
    shown = (*new["evidence_phases"], new["phase"], *old["evidence_phases"], old["phase"])
    phase = next((p for p in shown if p in STROKE.LABELS), None)
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


def analyze_attempt(video_path, session, progress=gr.Progress()):  # noqa: B008 - Gradio pattern
    """Outputs: status, card, checklist box, checklist, slow motion, key frames, comparison
    box, comparison text and image, session."""
    hidden, gone = gr.update(value=None, visible=False), gr.update(visible=False)
    empty = (gone, "", hidden, hidden, gone, "", None, session)
    if not video_path:
        return ("", notice_html("Falta el video", "Sube un video de tu saque o elige uno de la "
                                "demo, y pulsa Analizar saque."), *empty)  # fmt: skip
    result = analysis(video_path, progress)
    if "retake" in result:
        return (RECORDING_TIPS, notice_html("Necesitamos una nueva grabación", result["retake"]),
                *empty)  # fmt: skip
    if "error" in result:
        return ("", notice_html("No pudimos analizar el video", result["error"]), *empty)

    feedback, images = result["feedback"], result["images"]
    evidence = set(feedback["evidence_phases"])
    checks = feedback.get("checks", {})
    verdicts = {p: phase_verdict(checks, p) for p in STROKE.PHASES}
    gallery = [
        (rgb(framed(images[p], verdicts[p])),
         " ".join([STROKE.LABELS[p], MARKS[verdicts[p]][0] if verdicts[p] else "",
                   "★" if p in evidence else ""]).strip())
        for p in STROKE.PHASES
    ]  # fmt: skip
    current = {"feedback": feedback, "images": images, "record": history.attempt(feedback)}
    previous = session[-1] if session else None
    compare_text, compare_image = comparison(previous, current)
    return (
        "",
        result_html(feedback, result["seconds"], len(session) + 1, result.get("cached", False)),
        gr.update(visible=bool(checks)),
        checklist_markdown(checks) if checks else "",
        gr.update(value=result["slow"], visible=True),
        gr.update(value=gallery, visible=True),
        gr.update(visible=previous is not None),  # a comparison needs two attempts
        compare_text,
        compare_image,
        [*session, current],
    )


def start_over():
    """A new player at the stand: empty video, answers and session."""
    hidden, gone = gr.update(value=None, visible=False), gr.update(visible=False)
    return None, "", WELCOME, gone, "", hidden, hidden, gone, "", None, "", []


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
    gr.HTML(HEADER)
    session = gr.State([])
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            video_in = gr.Video(label="Video del saque", sources=["upload"])
            if DEMO_VIDEOS:
                gr.Examples([[str(p)] for p in DEMO_VIDEOS], inputs=[video_in],
                            label="Videos de la demo: elige uno y pulsa Analizar saque",
                            example_labels=[("Profesional" if p.stem.startswith("pro") else "Amateur")
                                            + f" · {p.stem}" for p in DEMO_VIDEOS])  # fmt: skip
            analyze_button = gr.Button("Analizar saque", variant="primary", size="lg")
            with gr.Row():
                restart_button = gr.Button("Empezar de nuevo")
                finish_button = gr.Button("Terminar sesión")
            player = gr.Textbox(
                label="ID de jugador (opcional, guarda tu historial)", placeholder="p01"
            )
            with gr.Accordion("Cómo grabar el saque", open=False):
                gr.Markdown(RECORDING_TIPS)
        with gr.Column(scale=2):
            card = gr.HTML(WELCOME)
            status = gr.Markdown()
            with gr.Accordion(
                "Qué revisamos en cada fase", open=False, visible=False
            ) as checks_box:
                checklist = gr.Markdown()
            slow_out = gr.Video(
                label="Cámara lenta con las fases", interactive=False, height=380, visible=False
            )
    gallery_out = gr.Gallery(
        label="Cuadros clave: verde = bien, rojo = a mejorar, gris = no se ve claro "
        "(★ = lo muestra la observación)",
        columns=6,
        height=230,
        object_fit="contain",
        visible=False,
    )
    with gr.Column(visible=False) as compare_box:
        comparison_text = gr.Markdown()
        comparison_image = gr.Image(label="Intento anterior y actual", interactive=False)
    summary_out = gr.Markdown()

    results = [status, card, checks_box, checklist, slow_out, gallery_out, compare_box,
               comparison_text, comparison_image, session]  # fmt: skip
    analyze_button.click(analyze_attempt, [video_in, session], results)
    restart_button.click(start_over, None, [video_in, *results[:-1], summary_out, session])
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
    demo.launch(server_name="127.0.0.1", max_file_size="200mb", theme=THEME, css=CSS)
