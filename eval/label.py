"""Labeling tool: each member labels the evaluation videos alone, then reviews the answers.

    uv run python -m eval.label      # then open http://127.0.0.1:7861

"Etiquetar": one video at a time, at normal speed and in slow motion, never with the system's
answers. Writes eval/serve/labels_<name>.csv (guide in data/README.md); the phase and the
drill follow from the category.
"Revisar respuestas": the key frame of each check in a run, with what the model said it saw.
Only videos the member already labeled are shown, so the labels stay blind. Writes
eval/serve/review_<name>.csv; `python -m eval.run_eval --review` turns it into the
hallucination rate.
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage data leaves the machine

import csv
import json
from pathlib import Path

import cv2
import gradio as gr

from core import ROOT, video
from core.analysis import crop, resize
from strokes import base

STROKE = base.get("serve")
EVAL = ROOT / "eval" / STROKE.NAME
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "processed" / "label"  # browser-playable copies of the videos
CATEGORIES = base.load(STROKE, "criteria.yaml")["categories"]
DRILLS = {d["category"]: d["id"] for d in base.load(STROKE, "drills.yaml")["drills"]}
NAMES = sorted(p.stem.removeprefix("labels_") for p in EVAL.glob("labels_*.csv"))
LABEL_COLUMNS = ["video", "player", "valid", "phase", "category", "drill_id", "notes"]
REVIEW_COLUMNS = ["run", "video", "category", "seen_ok", "answer_ok", "notes"]
OK = [("Sí", "yes"), ("No", "no"), ("No se puede saber", "unclear")]


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def write(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def playable(name: str, slowdown: int) -> str:
    """H.264 copy of a raw video (phones record HEVC, which browsers may not play)."""
    out = CACHE / f"{Path(name).stem}-x{slowdown}.mp4"
    if not out.exists():
        fps = video.probe(RAW / name).fps
        frames = (resize(frame, 1280) for _, frame in video.iter_frames(RAW / name))
        video.write_video(out, frames, fps / slowdown)
    return str(out)


def category_choices() -> list[tuple[str, str]]:
    return [(f"{STROKE.LABELS.get(c['phase'], 'Sin problema claro')} · {c['description']}", key)
            for key, c in CATEGORIES.items()]  # fmt: skip


def label_view(name: str, choice: str | None = None):
    rows = read(EVAL / f"labels_{name}.csv")
    done = sum(bool(r["valid"]) for r in rows)
    choice = choice or next((r["video"] for r in rows if not r["valid"]), rows[0]["video"])
    row = next(r for r in rows if r["video"] == choice)
    status = f"{done} de {len(rows)} videos etiquetados" + (
        " (todos)." if done == len(rows) else "."
    )
    return (
        gr.update(choices=[r["video"] for r in rows], value=choice),
        status,
        playable(choice, 1),
        playable(choice, 4),
        row["valid"] or None,
        row["category"] or None,
        row["notes"],
    )


def save_label(name: str, choice: str, valid: str | None, category: str | None, notes: str):
    if valid is None:
        raise gr.Error("Elige si el video se puede analizar.")
    if valid == "1" and not category:
        raise gr.Error("Elige el problema principal, o «sin problema claro».")
    path = EVAL / f"labels_{name}.csv"
    rows = read(path)
    row = next(r for r in rows if r["video"] == choice)
    keep = valid == "1"
    row.update(
        valid=valid,
        category=category if keep else "",
        phase=CATEGORIES[category]["phase"] if keep else "",
        drill_id=DRILLS[category] if keep else "",
        notes=notes.strip(),
    )
    write(path, rows, LABEL_COLUMNS)
    return label_view(name)  # the next video without a label


def runs() -> list[str]:
    """Runs of the full system (with checks), newest first."""
    files = sorted((EVAL / "runs").glob("*-full.jsonl"), reverse=True)
    return [p.stem for p in files if '"checks"' in p.read_text()]


def review_items(name: str, run: str) -> list[tuple[dict, str]]:
    """(record, category) of every check of the run, for videos `name` already labeled."""
    labeled = {r["video"] for r in read(EVAL / f"labels_{name}.csv") if r["valid"]}
    records = [json.loads(line) for line in (EVAL / "runs" / f"{run}.jsonl").open()]
    return [(record, category)
            for record in records if Path(record["video"]).name in labeled
            for category in (record.get("prediction") or {}).get("checks", {})]  # fmt: skip


def key_frame(record: dict, category: str):
    """The frame the model judged, cropped as the model saw it (box stored from v10 on)."""
    check = record["prediction"]["checks"][category]
    index = record["key_frames"][check.get("phase") or CATEGORIES[category]["phase"]]
    [(_, frame)] = video.sample(RAW / Path(record["video"]).name, 1, index, index)
    return cv2.cvtColor(resize(crop(frame, record.get("box")), 1024), cv2.COLOR_BGR2RGB)


def review_view(name: str, run: str | None):
    if not run:
        return None, "Elige una corrida.", None, None, "", None
    done = {(r["video"], r["category"]) for r in read(EVAL / f"review_{name}.csv")
            if r["run"] == run}  # fmt: skip
    items = review_items(name, run)
    left = [(r, c) for r, c in items if (Path(r["video"]).name, c) not in done]
    if not left:
        text = (f"No queda nada por revisar en {run}: {len(items)} revisiones hechas. "
                "Solo aparecen los videos que ya etiquetaste.")  # fmt: skip
        return None, text, None, None, "", None
    record, category = left[0]
    check = record["prediction"]["checks"][category]
    part, good, bad = CATEGORIES.get(category, {}).get("show", (category, "bien", "mal"))
    state = {"yes": good, "no": bad}.get(check["answer"], "no se ve claro")
    text = "\n\n".join([
        f"**{Path(record['video']).name}** · quedan {len(left)} de {len(items)}",
        f"Pregunta al modelo: _{CATEGORIES.get(category, {}).get('check', category)}_",
        f"El modelo dice que ve: **{check['seen']}**",
        f"Su respuesta: **{part}: {state}**",
    ])  # fmt: skip
    return key_frame(record, category), text, None, None, "", (Path(record["video"]).name, category)


def save_review(name, run, item, seen_ok, answer_ok, notes):
    if item is None:
        raise gr.Error("No hay nada que guardar.")
    if seen_ok is None or answer_ok is None:
        raise gr.Error("Responde las dos preguntas.")
    path = EVAL / f"review_{name}.csv"
    rows = [r for r in read(path) if (r["run"], r["video"], r["category"]) != (run, *item)]
    rows.append({"run": run, "video": item[0], "category": item[1], "seen_ok": seen_ok,
                 "answer_ok": answer_ok, "notes": notes.strip()})  # fmt: skip
    write(path, rows, REVIEW_COLUMNS)
    return review_view(name, run)


with gr.Blocks(title="Serve Lab: etiquetas", analytics_enabled=False) as demo:
    gr.Markdown("# Etiquetas del set de evaluación\nGuía: data/README.md. Cada uno etiqueta solo.")
    name = gr.Radio(NAMES, value=NAMES[0], label="Quién etiqueta")
    with gr.Tab("Etiquetar"):
        gr.Markdown("Mira el saque a velocidad normal y en cámara lenta. Juzga solo con las "
                    "descripciones de las categorías; el estilo personal no es un error.")  # fmt: skip
        with gr.Row():
            choice = gr.Dropdown(label="Video")
            status = gr.Markdown()
        with gr.Row():
            normal = gr.Video(label="Velocidad normal", interactive=False)
            slow = gr.Video(label="Cámara lenta (x4)", interactive=False)
        valid = gr.Radio([("Se puede analizar", "1"), ("Pedir otra grabación", "0")],
                         label="¿Una persona podría juzgar este saque?")  # fmt: skip
        category = gr.Radio(category_choices(), label="Problema más claro (si hay dos, el de "
                            "la fase más temprana)")  # fmt: skip
        notes = gr.Textbox(label="Notas (opcional)")
        save = gr.Button("Guardar y siguiente", variant="primary")
    with gr.Tab("Revisar respuestas"):
        gr.Markdown("Después de etiquetar: ¿lo que dijo el modelo coincide con el cuadro?")
        run = gr.Dropdown(runs(), label="Corrida (eval/serve/runs)")
        item = gr.State()
        with gr.Row():
            frame = gr.Image(label="Cuadro que juzgó el modelo", interactive=False)
            with gr.Column():
                question = gr.Markdown()
                seen_ok = gr.Radio(OK, label="¿Lo que dice que ve está en el cuadro?")
                answer_ok = gr.Radio(OK, label="¿Su respuesta es correcta?")
                review_notes = gr.Textbox(label="Notas (opcional)")
                save_check = gr.Button("Guardar y siguiente", variant="primary")

    label_out = [choice, status, normal, slow, valid, category, notes]
    review_out = [frame, question, seen_ok, answer_ok, review_notes, item]
    demo.load(label_view, [name], label_out)
    name.input(label_view, [name], label_out).then(review_view, [name, run], review_out)
    choice.input(label_view, [name, choice], label_out)
    save.click(save_label, [name, choice, valid, category, notes], label_out)
    run.input(review_view, [name, run], review_out)
    save_check.click(save_review, [name, run, item, seen_ok, answer_ok, review_notes], review_out)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
