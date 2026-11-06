"""Labeling tool: each member labels the evaluation videos alone, then reviews the answers.

    uv run python -m eval.label      # then open http://127.0.0.1:7861

"Etiquetar": one video at a time, at normal speed and in slow motion next to a professional
serve, never with the system's answers. Two yes/no questions that need no tennis knowledge: can
the serve be judged, and has the tossing hand already dropped to the shoulder when the racket is
lowest behind the back (toss_arm_drops_early in criteria.yaml). Writes
eval/serve/labels_<name>.csv (guide in data/README.md); the phase and the drill follow from the
answer.
"Revisar respuestas": the key frame of each check in the runs of the final Project 3 version
(REVIEW_RUNS), with what the model said it saw and concluded; the member says whether both match
the frame. Only videos the member already labeled are shown, so the labels stay blind. Writes
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
REFERENCE = "pro02_s01.mp4"  # a professional filmed from the side, shown next to every video
# The final Project 3 version is v22: v21 plus a retry that only fires after three guesses in a
# row. With seed 0, the app's, v21 never guessed three times, so these v21 runs are v22's answers.
REVIEW_RUNS = [("JC: 21 saques", "20261102-131618-full"),
               ("Profesionales: 11 saques", "20261102-132154-full")]  # fmt: skip
ARM = [("Sí, ya está a la altura del hombro o más abajo", "toss_arm_drops_early"),
       ("No, sigue arriba del hombro, apuntando a la pelota", "no_clear_problem")]  # fmt: skip


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
        raise gr.Error("Responde si la mano que lanzó ya bajó.")
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


def pending(name: str, run: str) -> tuple[list[tuple[dict, str]], int]:
    """The checks of the run `name` has not reviewed yet, and how many there are in all."""
    done = {(r["video"], r["category"]) for r in read(EVAL / f"review_{name}.csv")
            if r["run"] == run}  # fmt: skip
    items = review_items(name, run)
    return [(r, c) for r, c in items if (Path(r["video"]).name, c) not in done], len(items)


def review_view(name: str, run: str | None):
    """Frame, text, empty answers, the check on screen, and whether the answer form shows."""
    if not run:
        return None, "Elige una corrida.", None, None, "", None, gr.update(visible=False)
    left, total = pending(name, run)
    if not left:
        others = [label for label, other in REVIEW_RUNS if other != run and pending(name, other)[0]]
        text = (f"### Terminaste esta corrida: revisaste las {total} respuestas.\n\n"
                + (f"Falta otra: elige «{others[0]}» en la lista de arriba." if others else
                   "Ya revisaste todas las corridas. Aquí no queda nada por hacer.")
                if total else "No hay nada que revisar: primero etiqueta los videos en "
                "«Etiquetar».")  # fmt: skip
        return None, text, None, None, "", None, gr.update(visible=False)
    record, category = left[0]
    check = record["prediction"]["checks"][category]
    part, good, bad = CATEGORIES[category]["show"]
    state = {"yes": good, "no": bad}.get(check["answer"], "no se ve claro")
    phase = STROKE.LABELS[check.get("phase") or CATEGORIES[category]["phase"]]
    text = "\n\n".join([
        f"### Respuesta {total - len(left) + 1} de {total}",
        f"{Path(record['video']).name} · cuadro de la fase {phase}",
        f"Qué revisaba el modelo: **{part}** (bien: {good}; mal: {bad})",
        f"Lo que el modelo dice que ve (en inglés): _{check['seen']}_",
        f"Su conclusión: **{part}: {state}**",
    ])  # fmt: skip
    item = (Path(record["video"]).name, category)
    return key_frame(record, category), text, None, None, "", item, gr.update(visible=True)


def restore(saved: list[str]) -> tuple[str, str]:
    """The person and run this browser chose last, so a reload never opens someone else's file."""
    who, chosen = saved
    runs = [other for _, other in REVIEW_RUNS]
    return who if who in NAMES else NAMES[0], chosen if chosen in runs else runs[0]


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
    # Kept in the browser: the name and run are not private, and a fixed secret survives restarts
    remembered = gr.BrowserState([NAMES[0], REVIEW_RUNS[0][1]], storage_key="serve-lab-labels",
                                 secret="serve-lab-labels")  # fmt: skip
    with gr.Tab("Etiquetar"):
        gr.Markdown("Mira el saque en cámara lenta, al lado del saque profesional de "
                    "referencia. Responde solo lo que ves; no hace falta saber de tenis.")  # fmt: skip
        with gr.Row():
            choice = gr.Dropdown(label="Video")
            status = gr.Markdown()
        with gr.Row():
            normal = gr.Video(label="Velocidad normal", interactive=False)
            slow = gr.Video(label="Cámara lenta (x4)", interactive=False)
            reference = gr.Video(label="Referencia: profesional en cámara lenta (x4)",
                                 interactive=False)  # fmt: skip
        valid = gr.Radio([("Sí", "1"), ("No, pedir otra grabación", "0")],
                         label="1. ¿Se ve al jugador de lado y de cuerpo entero, con la mano "
                         "que lanza la pelota visible?")  # fmt: skip
        category = gr.Radio(ARM, label="2. En el momento en que la raqueta está más abajo, "
                            "detrás de la espalda (justo antes de subir a pegarle), ¿la mano que "
                            "lanzó la pelota ya bajó?")  # fmt: skip
        notes = gr.Textbox(label="Notas (opcional)")
        save = gr.Button("Guardar y siguiente", variant="primary")
    with gr.Tab("Revisar respuestas"):
        gr.Markdown(
            "Aquí se mide si el modelo de Qwen (proyecto 3) inventa cosas. Cada pantalla es una "
            "respuesta del modelo: la imagen que miró, lo que dice que ve y su conclusión. Mira "
            "**solo la imagen** y responde:\n\n"
            "1. ¿Lo que dice que ve aparece de verdad en la imagen? «No» si describe algo que no "
            "está, por ejemplo, dice que el brazo está arriba y está abajo.\n"
            "2. ¿Su conclusión es correcta para esta imagen?\n\n"
            "«No se puede saber» si la imagen no deja ver esa parte del cuerpo. Cada respuesta se "
            "guarda al pulsar el botón: puedes cerrar y seguir después. Hay que revisar las dos "
            "corridas de la lista; al terminar una, la pantalla lo dice."
        )
        run = gr.Dropdown(REVIEW_RUNS, value=REVIEW_RUNS[0][1], label="Corrida a revisar")
        # The check on screen lives in the page, not in a server session that a restart forgets
        item = gr.JSON(visible="hidden")
        with gr.Row():
            frame = gr.Image(label="Imagen que miró el modelo", interactive=False)
            with gr.Column():
                question = gr.Markdown()
                with gr.Column(visible=False) as form:
                    seen_ok = gr.Radio(OK, label="1. ¿Lo que dice que ve aparece en la imagen?")
                    answer_ok = gr.Radio(OK, label="2. ¿Su conclusión es correcta?")
                    review_notes = gr.Textbox(label="Notas (opcional)")
                    save_check = gr.Button("Guardar y siguiente", variant="primary")

    label_out = [choice, status, normal, slow, valid, category, notes]
    review_out = [frame, question, seen_ok, answer_ok, review_notes, item, form]
    remember = lambda who, chosen: [who, chosen]
    demo.load(restore, remembered, [name, run]).then(label_view, [name], label_out).then(
        review_view, [name, run], review_out)  # fmt: skip
    demo.load(lambda: playable(REFERENCE, 4), None, reference)
    name.input(label_view, [name], label_out).then(review_view, [name, run], review_out).then(
        remember, [name, run], remembered)  # fmt: skip
    choice.input(label_view, [name, choice], label_out)
    save.click(save_label, [name, choice, valid, category, notes], label_out)
    run.input(review_view, [name, run], review_out).then(remember, [name, run], remembered)
    save_check.click(save_review, [name, run, item, seen_ok, answer_ok, review_notes], review_out)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
