"""Labeling tool: each member labels the evaluation videos alone, then reviews the answers.

    uv run python -m eval.label      # then open http://127.0.0.1:7861

"Etiquetar": one video at a time, at normal speed and in slow motion next to a professional
serve, never with the system's answers. Two yes/no questions that need no tennis knowledge: can
the serve be judged, and has the tossing hand already dropped to the shoulder when the racket is
lowest behind the back (toss_arm_drops_early in criteria.yaml). Writes
eval/serve/labels_<name>.csv (guide in data/README.md); the phase and the drill follow from the
answer.
"Cuerpo": the rest of the serve for the judged videos of labels.csv, one multiple-choice question
per part (ASPECTS: legs, shoulders, the racket when the ball is highest and behind the back, the
arm and the contact point at impact, the landing), each with what to look at and the ideal. The
players come in turn, so any number of answers covers all of them. Writes
eval/serve/aspects_<name>.csv.
"Decidir juntos": the videos where the members' answers differ, with each member's answer, for
the members to watch together and answer once. Writes eval/serve/labels.csv: the rows everyone
agrees on and the decisions taken so far.
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
import re
from contextlib import contextmanager
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
AGREED = EVAL / "labels.csv"  # what strokes/serve/dataset.py trains on
LABEL_COLUMNS = ["video", "player", "valid", "phase", "category", "drill_id", "notes"]
REVIEW_COLUMNS = ["run", "video", "category", "seen_ok", "answer_ok", "notes"]
OK = [("Sí", "yes"), ("No", "no"), ("No se puede saber", "unclear")]
REFERENCE = "pro02_s01.mp4"  # a professional filmed from the side, shown next to every video
# The final Project 3 version is v22: v21 plus a retry that only fires after three guesses in a
# row. With seed 0, the app's, v21 never guessed three times, so these v21 runs are v22's answers.
REVIEW_RUNS = [("JC: 21 saques", "20261102-131618-full"),
               ("Profesionales: 11 saques", "20261102-132154-full")]  # fmt: skip
VALID = [("Sí", "1"), ("No, pedir otra grabación", "0")]
ARM = [("Sí, ya está a la altura del hombro o más abajo", "toss_arm_drops_early"),
       ("No, sigue arriba del hombro, apuntando a la pelota", "no_clear_problem")]  # fmt: skip
ASK_VALID = ("1. ¿Se ve al jugador de lado y de cuerpo entero, con la mano que lanza la pelota "
             "visible?")  # fmt: skip
ASK_ARM = ("2. En el momento en que la raqueta está más abajo, detrás de la espalda (justo antes "
           "de subir a pegarle), ¿la mano que lanzó la pelota ya bajó?")  # fmt: skip
# "Cuerpo": (column, question, what to look at and the ideal, options); "unseen" when it cannot be seen
DEGREE = [("No, casi nada", "none"), ("Poco, hace falta mejorar", "little"), ("Sí, bien", "good")]
ASPECTS = [
    ("legs", "1. Piernas: antes de subir a pegarle, ¿dobla las rodillas?",
     ("Míralo cuando la raqueta va detrás de la espalda. Lo ideal: rodillas dobladas, la cadera "
      "baja como para saltar."),
     DEGREE + [("Demasiado, se agacha mucho", "too_much"), ("No se ve", "unseen")]),
    ("shoulders", ("2. Cuerpo: con la mano que lanza arriba, ¿el hombro de esa mano queda más alto "
                   "que el otro (el cuerpo se inclina hacia atrás)?"),
     "Lo ideal: el hombro de la mano que lanza claramente más alto, el cuerpo un poco arqueado.",
     DEGREE + [("Demasiado, se va hacia atrás", "too_much"), ("No se ve", "unseen")]),
    ("racket_at_peak", "3. Raqueta: cuando la pelota está en lo más alto, ¿dónde está la raqueta?",
     ("Lo ideal: arriba detrás de la cabeza, con el codo a la altura del hombro (posición de "
      "trofeo); cae detrás de la espalda cuando la pelota empieza a bajar."),
     [("Todavía abajo, cerca de la cadera", "low"), ("Subiendo, a la altura del hombro", "rising"),
      ("Arriba detrás de la cabeza, codo a la altura del hombro", "trophy"),
      ("Ya cayó detrás de la espalda", "dropped"), ("Ya está golpeando la pelota", "hitting"),
      ("No se ve la pelota o la raqueta", "unseen")]),
    ("racket_drop", ("4. Raqueta: antes de subir a pegarle, ¿cae detrás de la espalda, con la "
                     "punta hacia el suelo?"),
     "Lo ideal: sí, como si se rascara la espalda; luego sube hacia la pelota.",
     [("No, va directo a la pelota", "none"), ("Poco, se queda a medio camino", "little"),
      ("Sí, bien", "good"), ("No se ve", "unseen")]),
    ("arm_at_contact", "5. Impacto: al pegarle, ¿el brazo de la raqueta está estirado hacia arriba?",
     "Lo ideal: el brazo estirado casi vertical, pegándole lo más alto que alcanza.",
     [("No, doblado o a la altura de la cabeza", "bent"),
      ("Estirado, pero hacia adelante, no arriba", "forward"),
      ("Sí, estirado hacia arriba", "up"), ("No se ve", "unseen")]),
    ("contact_point", "6. Impacto: ¿dónde le pega a la pelota respecto al cuerpo?",
     "Fíjate hacia dónde está la cancha. Lo ideal: un poco adelante del cuerpo, hacia la cancha.",
     [("Detrás de la cabeza", "behind"), ("Justo encima de la cabeza", "above"),
      ("Un poco adelante", "ahead"),
      ("Muy adelante, tiene que inclinarse para alcanzarla", "far_ahead"), ("No se ve", "unseen")]),
    ("landing", "7. Final: después de pegarle, ¿cómo cae?",
     "Lo ideal: hacia adelante, dentro de la cancha, sobre el pie de adelante y en equilibrio.",
     [("Hacia adelante, en equilibrio", "forward"), ("En el mismo lugar", "in_place"),
      ("Hacia atrás o de lado, pierde el equilibrio", "off_balance"), ("No se ve", "unseen")]),
]  # fmt: skip
ASPECT_COLUMNS = ["video", "player", *(a[0] for a in ASPECTS), "notes"]
# The videos stay in view while the member scrolls through the questions next to them; Gradio's
# container hides its overflow, which would turn it into the scroll box that `sticky` follows.
CSS = """.gradio-container { overflow-x: clip !important; overflow-y: visible !important; }
.watch { position: sticky; top: 8px; align-self: flex-start; }"""


@contextmanager
def beside_videos():
    """The serve in slow motion, at normal speed and the professional's on the left, and the
    questions written inside the `with` block on the right."""
    with gr.Row():
        with gr.Column(scale=3, elem_classes="watch"):
            # Heights fit a laptop screen, with vertical phone videos too
            slow = gr.Video(label="Cámara lenta (x4)", interactive=False, height=420)
            with gr.Row():
                normal = gr.Video(label="Velocidad normal", interactive=False, height=220)
                reference = gr.Video(label="Referencia: profesional en cámara lenta (x4)",
                                     interactive=False, height=220)  # fmt: skip
        with gr.Column(scale=2):
            yield normal, slow, reference


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


def rows_of(name: str) -> list[dict]:
    """The member's rows, plus an empty one for each video in data/raw/ not in the file yet (new
    clips, for example from strokes/serve/split.py) nor in labels.csv (labeled by someone else at
    the team's request); saving a label writes them to the file."""
    rows = read(EVAL / f"labels_{name}.csv")
    listed = {r["video"] for r in rows} | {r["video"] for r in read(AGREED)}
    new = sorted(p.name for p in RAW.iterdir()
                 if p.suffix.lower() in (".mp4", ".mov") and p.name not in listed)  # fmt: skip
    empty = dict.fromkeys(LABEL_COLUMNS, "")
    return rows + [{**empty, "video": v, "player": re.split(r"[-_]", Path(v).stem)[0]} for v in new]


def label_view(name: str, choice: str | None = None):
    """The videos still to label (every video once all are labeled) and `choice` or the next."""
    rows = rows_of(name)
    left = [r["video"] for r in rows if not r["valid"]]
    choice = choice or (left[0] if left else rows[0]["video"])
    row = next(r for r in rows if r["video"] == choice)
    status = (f"Faltan {len(left)} de {len(rows)} videos; la lista muestra solo esos." if left
              else f"Los {len(rows)} videos están etiquetados.")  # fmt: skip
    return (
        gr.update(choices=left or [r["video"] for r in rows], value=choice),
        status,
        playable(choice, 1),
        playable(choice, 4),
        row["valid"] or None,
        row["category"] or None,
        row["notes"],
    )


def answered(row: dict, valid: str | None, category: str | None, notes: str) -> dict:
    """`row` with the two answers, and the phase and drill that follow from them."""
    if valid is None:
        raise gr.Error("Elige si el video se puede analizar.")
    if valid == "1" and not category:
        raise gr.Error("Responde si la mano que lanzó ya bajó.")
    keep = valid == "1"
    return {
        **row,
        "valid": valid,
        "category": category if keep else "",
        "phase": CATEGORIES[category]["phase"] if keep else "",
        "drill_id": DRILLS[category] if keep else "",
        "notes": notes.strip(),
    }


def save_label(name: str, choice: str, valid: str | None, category: str | None, notes: str):
    path = EVAL / f"labels_{name}.csv"
    rows = [
        answered(r, valid, category, notes) if r["video"] == choice else r for r in rows_of(name)
    ]
    write(path, rows, LABEL_COLUMNS)
    return label_view(name)  # the next video without a label


def aspect_rows(name: str) -> list[dict]:
    """The member's answers about the body, one row per judged video of labels.csv, taking the
    players in turn (JC after the feedback apart) so that a partial pass covers all of them."""
    saved = {r["video"]: r for r in read(EVAL / f"aspects_{name}.csv")}
    empty = dict.fromkeys(ASPECT_COLUMNS, "")
    turns, ranked = {}, []
    for r in read(AGREED):
        if r["valid"] == "1":
            group = "jc-post" if r["video"].startswith("jc-post") else r["player"]
            turns[group] = turns.get(group, 0) + 1
            row = saved.get(r["video"], {**empty, "video": r["video"], "player": r["player"]})
            ranked.append((turns[group], group, row))
    return [row for *_, row in sorted(ranked, key=lambda t: t[:2])]


def aspect_view(name: str, choice: str | None = None):
    """The videos still to answer (every video once all are answered), `choice` or the next, and
    its answers."""
    rows = aspect_rows(name)
    left = [r["video"] for r in rows if not r[ASPECTS[0][0]]]
    choice = choice or (left[0] if left else rows[0]["video"])
    row = next(r for r in rows if r["video"] == choice)
    status = (f"Faltan {len(left)} de {len(rows)} videos. No hace falta hacerlos todos: van "
              "alternando jugadores, así que cualquier avance sirve." if left
              else f"Los {len(rows)} videos tienen respuesta.")  # fmt: skip
    answers = [row[a[0]] or None for a in ASPECTS]
    return (gr.update(choices=left or [r["video"] for r in rows], value=choice), status,
            playable(choice, 1), playable(choice, 4), *answers, row["notes"])  # fmt: skip


def save_aspects(name: str, choice: str, *answers):
    *answers, notes = answers
    if None in answers:
        raise gr.Error("Responde todas las preguntas; si algo no se ve, elige «No se ve».")
    done = dict(zip((a[0] for a in ASPECTS), answers))
    rows = [{**r, **done, "notes": notes.strip()} if r["video"] == choice else r
            for r in aspect_rows(name)]  # fmt: skip
    write(EVAL / f"aspects_{name}.csv", [r for r in rows if r[ASPECTS[0][0]]], ASPECT_COLUMNS)
    return aspect_view(name)  # the next video without answers


def disputes() -> tuple[list[dict], dict[str, list[dict]]]:
    """The rows every member labeled the same, and each member's row of the videos that differ."""
    files = [{r["video"]: r for r in read(EVAL / f"labels_{name}.csv")} for name in NAMES]
    agreed, differ = [], {}
    for video_name, first in files[0].items():
        rows = [f.get(video_name) or {"valid": ""} for f in files]
        if not all(r["valid"] for r in rows):
            continue  # someone has not labeled it yet
        if len({(r["valid"], r["category"]) for r in rows}) > 1:
            differ[video_name] = rows
        else:
            notes = "; ".join(f"{n}: {r['notes']}" for n, r in zip(NAMES, rows) if r["notes"])
            agreed.append({**first, "notes": notes})
    return agreed, differ


def agree_view(choice: str | None = None):
    """The next video still to decide (or `choice`), with each member's answer."""
    _, differ = disputes()
    if not differ:
        text = "No hay desacuerdos: primero etiqueten los videos cada uno en «Etiquetar»."
        return gr.update(choices=[], value=None), text, None, None, "", None, None, ""
    decided = {r["video"]: r for r in read(AGREED) if r["video"] in differ}
    left = [v for v in differ if v not in decided]
    choice = choice or (left[0] if left else next(iter(differ)))
    if left:
        status = f"### Decididos {len(decided)} de {len(differ)}"
    else:
        status = (f"### Listo: decidieron los {len(differ)} videos.\n\n`eval/serve/labels.csv` "
                  "está completo. Para cambiar una decisión, elijan el video en la lista.")  # fmt: skip
    said = []
    for name, row in zip(NAMES, differ[choice]):
        answer = {v: k for k, v in ARM}.get(row["category"], "no se puede analizar")
        said.append(f"- **{name.capitalize()}**: {answer}" + (f" (nota: {row['notes']})"
                    if row["notes"] else ""))  # fmt: skip
    row = decided.get(choice, {})
    return (
        gr.update(choices=list(differ), value=choice),
        status,
        playable(choice, 1),
        playable(choice, 4),
        "Lo que respondió cada uno:\n\n" + "\n".join(said),
        row.get("valid") or None,
        row.get("category") or None,
        row.get("notes", ""),
    )


def save_agreement(choice: str, valid: str | None, category: str | None, notes: str):
    agreed, differ = disputes()
    decided = {r["video"]: r for r in read(AGREED) if r["video"] in differ}
    decided[choice] = answered(differ[choice][0], valid, category, notes)
    order = {r["video"]: i for i, r in enumerate(read(EVAL / f"labels_{NAMES[0]}.csv"))}
    rows = sorted(agreed + list(decided.values()), key=lambda r: order[r["video"]])
    others = [r for r in read(AGREED) if r["video"] not in order]  # not the members' labels
    write(AGREED, rows + others, LABEL_COLUMNS)
    return agree_view()  # the next video still to decide


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
        with beside_videos() as (normal, slow, reference):
            valid = gr.Radio(VALID, label=ASK_VALID)
            category = gr.Radio(ARM, label=ASK_ARM)
            notes = gr.Textbox(label="Notas (opcional)")
            save = gr.Button("Guardar y siguiente", variant="primary")
    with gr.Tab("Cuerpo"):
        gr.Markdown("Siete preguntas sobre el resto del saque: piernas, cuerpo, raqueta, impacto "
                    "y final. Compáralo con el saque profesional de referencia. Responde lo que "
                    "ves; debajo de cada pregunta dice qué mirar y cómo sería lo ideal. Si algo "
                    "no se ve, elige «No se ve».")  # fmt: skip
        with gr.Row():
            aspect_choice = gr.Dropdown(label="Video")
            aspect_status = gr.Markdown()
        with beside_videos() as (aspect_normal, aspect_slow, aspect_reference):
            aspects = [gr.Radio(options, label=ask, info=info) for _, ask, info, options in ASPECTS]
            aspect_notes = gr.Textbox(label="Notas (opcional)")
            save_aspect = gr.Button("Guardar y siguiente", variant="primary")
    with gr.Tab("Decidir juntos"):
        gr.Markdown("Aquí salen solo los videos en que sus respuestas no coinciden. Véanlos "
                    "juntos, comenten qué ve cada uno y respondan una sola vez entre los dos. "
                    "Decidan solo con lo que se ve en el video.")  # fmt: skip
        with gr.Row():
            agree_choice = gr.Dropdown(label="Video")
            agree_status = gr.Markdown()
        with beside_videos() as (agree_normal, agree_slow, agree_reference):
            said = gr.Markdown()
            agree_valid = gr.Radio(VALID, label=ASK_VALID)
            agree_category = gr.Radio(ARM, label=ASK_ARM)
            agree_notes = gr.Textbox(label="Por qué lo decidieron (opcional)")
            save_agree = gr.Button("Guardar decisión y siguiente", variant="primary")
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
    agree_out = [agree_choice, agree_status, agree_normal, agree_slow, said, agree_valid,
                 agree_category, agree_notes]  # fmt: skip
    aspect_out = [aspect_choice, aspect_status, aspect_normal, aspect_slow, *aspects, aspect_notes]
    remember = lambda who, chosen: [who, chosen]
    demo.load(restore, remembered, [name, run]).then(label_view, [name], label_out).then(
        review_view, [name, run], review_out).then(aspect_view, [name], aspect_out)  # fmt: skip
    name.input(aspect_view, [name], aspect_out)
    aspect_choice.input(aspect_view, [name, aspect_choice], aspect_out)
    save_aspect.click(save_aspects, [name, aspect_choice, *aspects, aspect_notes], aspect_out)
    demo.load(lambda: playable(REFERENCE, 4), None, reference)
    demo.load(lambda: playable(REFERENCE, 4), None, agree_reference)
    demo.load(lambda: playable(REFERENCE, 4), None, aspect_reference)
    demo.load(agree_view, None, agree_out)
    name.input(label_view, [name], label_out).then(review_view, [name, run], review_out).then(
        remember, [name, run], remembered)  # fmt: skip
    choice.input(label_view, [name, choice], label_out)
    save.click(save_label, [name, choice, valid, category, notes], label_out)
    agree_choice.input(agree_view, [agree_choice], agree_out)
    save_agree.click(save_agreement, [agree_choice, agree_valid, agree_category, agree_notes],
                     agree_out)  # fmt: skip
    run.input(review_view, [name, run], review_out).then(remember, [name, run], remembered)
    save_check.click(save_review, [name, run, item, seen_ok, answer_ok, review_notes], review_out)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, css=CSS)
