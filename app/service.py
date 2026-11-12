"""What the Serve Lab app runs: one analysis of a video with the team's own model, and the demo
cache. Used by the web app's API (app/api.py).

One analysis: MediaPipe finds the body in every frame (core.pose), the serve model says whether
the tossing arm drops early and which joints and moments the answer rests on
(strokes/serve/model.py), and the frames around contact become key moments and a slow motion with
the skeleton drawn. The answer is a list of up to MAX_POINTS points to improve (points): the
tossing arm and the aspects of the body in experiments/serve/points.csv (eval/body.py --levels),
each with how far it can be trusted (LEVELS) and its drill. The toss height and the body measures
that separate amateurs from professionals are shown next to the professionals'. No language model
and no internet.
"""

import csv
import hashlib
import json
import shutil
import sys
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from core import ROOT, load_config, pose, video
from core.analysis import InsufficientEvidence, check_video, label, resize
from strokes import base
from strokes.serve import body, dataset, model

CONFIG = load_config()
STROKE = base.get("serve")
DRILLS = {d["id"]: d for d in base.load(STROKE, "drills.yaml")["drills"]}
MODEL = model.load(CONFIG["serve"]["model"])
WEIGHTS = (model.baseline.RESULTS / f"{MODEL['name']}.pt").read_bytes()
CONFIDENCE = {"high": "alta", "medium": "media", "low": "baja"}
OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="servelab-"))
MIN_FOUND = 0.5  # share of frames with a person, under which the video is recorded again
# Share of arm points MediaPipe saw around contact. On the train split the 7 serves the team could
# not judge saw 0.58 to 0.82 and the others 0.77 or more: under MIN_SEEN (3 of the 7, no judgeable
# one) the video is recorded again, under SURE_SEEN the answer is at most of low confidence.
MIN_SEEN, SURE_SEEN = 0.7, 0.85
SURE_MARGIN = 0.15  # distance of the probability to the threshold for a high confidence
SMALL_FALL = 0.02  # an occlusion fall under it moves the answer too little to name a reason
IMAGE_SIDE = 720  # longest side of the key moments and the slow motion
# Joints of the app model's occlusion map (model.PARTS or model.HEIGHT_PARTS) in Spanish
SPANISH = {"tossing": "lanza", "racket": "raqueta", "shoulder": "Hombro", "elbow": "Codo",
           "wrist": "Muñeca"}  # fmt: skip
PARTS = [f"{SPANISH[joint]} ({SPANISH[arm]})" for arm, joint in map(str.split, MODEL["parts"])]
# The professionals' toss height (eval/improvement.py), and what the app says about each aspect of
# the body (eval/body.py --levels): a point to improve from a body model or from its measure
# against the professionals' range, or the measure next to the professionals' and no point
_RESULTS = model.baseline.RESULTS
PROS_TOSS = {r["measure"]: float(r["value"]) for r in csv.DictReader((_RESULTS / "improvement.csv").open())
             if r["group"] == "professionals"}  # fmt: skip
POINTS = {r["aspect"]: r for r in csv.DictReader((_RESULTS / "points.csv").open())
          if r["level"] or r["described"]}  # fmt: skip
BODY = {name: body.load(name) for name in {r["model"] for r in POINTS.values()} - {""}}
BODY_WEIGHTS = b"".join((_RESULTS / name).read_bytes()
                        for name in ["points.csv", *(f"{m}.pt" for m in sorted(BODY))])  # fmt: skip
LEVELS = {"validated": "Modelo validado", "trial": "Modelo en prueba",
          "measure": "Medida frente a profesionales"}  # fmt: skip
RANK = {"validated": 2, "trial": 1, "measure": 0}  # the order of the points: the surest first
MAX_POINTS = 3  # points to improve one analysis gives at most
# Each aspect in the words of the "Cuerpo" form: its audited measure (body.measures: higher means
# more to improve; negated here it is a distance where more is better) and its drill
BODY_TEXT = {
    "legs": ("Piernas", "la cadera baja {:.2f} torsos antes del salto", "knee_load"),
    "racket_drop": ("Caída de la raqueta", ("antes de subir a pegarle, la muñeca de la raqueta "
                                             "cae {:.2f} torsos"), None),
    "arm_at_contact": ("Brazo al impacto", ("al impacto, la muñeca de la raqueta llega {:.2f} "
                                             "torsos sobre su hombro"), "high_reach"),
    "landing": ("Aterrizaje", "después del impacto, la cadera avanza {:.2f} torsos", "land_and_hold"),
}  # fmt: skip
LEGS = ("legs", "landing")  # aspects read from the legs: they need them seen
ANSWER = 3  # what a saved answer holds (3: the body points); a new one makes the demo cache again
MOMENTS = {"start": "Inicio", "arm_up": "Brazo más arriba", "arm_down": "La muñeca baja del hombro",
           "contact": "Impacto"}  # fmt: skip

# Demo mode (uv run python -m app.api --demo): the videos in data/demo/ are analyzed once at start
# and their answers saved in data/processed/demo_cache/, so at the presentation they show at once.
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


def confidence(answer: dict) -> str:
    """high when every seed agrees, MediaPipe saw the arms and the probability is far from the
    threshold; low when the seeds disagree or many arm points were not seen."""
    agree = all((s >= answer["threshold"]) == answer["drops_early"] for s in answer["seeds"])
    if not agree or answer["seen"] < SURE_SEEN:
        return "low"
    return "high" if abs(answer["probability"] - answer["threshold"]) >= SURE_MARGIN else "medium"


def certainty(answer: dict) -> str:
    """How far the answer can be trusted, in Spanish: seeds, points seen and test rates."""
    side = [(s >= answer["threshold"]) == answer["drops_early"] for s in answer["seeds"]]
    return (f"{sum(side)} de {len(side)} versiones del modelo coinciden y MediaPipe vio el "
            f"{answer['seen']:.0%} de los puntos de tus brazos. En los videos de prueba, el "
            f"modelo encontró el problema en {MODEL['test_tpr']:.0%} de los saques que lo tenían "
            f"y recomendó el ejercicio de más en {MODEL['test_fpr']:.0%} de los que no.")  # fmt: skip


def observation(answer: dict) -> str:
    """What the landmarks show, with B0's measure: when the tossing wrist goes below its
    shoulder."""
    if answer["drop_seconds"] > dataset.BEFORE / dataset.FPS:
        return ("En los 1.6 s antes del impacto no vemos tu muñeca de lanzamiento arriba del "
                "hombro.")  # fmt: skip
    if answer["drop_seconds"] == 0:
        return "Tu muñeca de lanzamiento sigue arriba del hombro en el momento del impacto."
    return f"Tu muñeca de lanzamiento baja del hombro {answer['drop_seconds']:.2f} s antes del impacto."


def reason(answer: dict) -> str:
    """The joint and stretch that move the answer most (occlusion), in Spanish."""
    fall = np.array(answer["occlusion"])
    j, k = np.unravel_index(fall.argmax() if answer["drops_early"] else fall.argmin(), fall.shape)
    if abs(fall[j, k]) < SMALL_FALL:
        return ("Ningún tramo por sí solo cambia mucho la respuesta: el modelo mira el movimiento "
                "completo del brazo.")  # fmt: skip
    start = model.seconds(model.STARTS[k])
    when = stretch_text(start, start + model.BIN / dataset.FPS)
    if answer["drops_early"]:
        return f"Lo que más pesa hacia «baja antes de tiempo»: {PARTS[j].lower()} {when}."
    return f"Lo que más se parece a un saque sin el problema: {PARTS[j].lower()} {when}."


def toss_progress(answer: dict) -> dict:
    """The serve's toss height against the professionals' median and 10th percentile."""
    height = answer["toss_height"]
    median = PROS_TOSS["toss height (median, torso lengths)"]
    floor = PROS_TOSS["toss height (10th percentile, torso lengths)"]
    return {
        "height": round(height, 2), "pros_median": round(median, 2), "pros_floor": round(floor, 2),
        "in_range": height >= floor,
        "text": f"Cuando la raqueta está más baja, tu mano de lanzamiento queda {height:.2f} torsos "
                f"sobre el hombro. En los profesionales, la mediana es {median:.2f} y 9 de cada 10 "
                f"llegan a {floor:.2f} o más.",
    }  # fmt: skip


def measured(aspect: str, answer: dict) -> tuple[str, str]:
    """An aspect's measure of the serve and the professionals', in Spanish."""
    _, sentence, _ = BODY_TEXT[aspect]
    row = POINTS[aspect]
    pros = (f"En los profesionales, la mediana es {-float(row['pros_median']):.2f} y 9 de cada 10 "
            f"llegan a {-float(row['pros_p90']):.2f} o más.")  # fmt: skip
    value = -answer["body"][aspect]
    if aspect == "landing" and value < 0:  # the hips end behind where they were at contact
        sentence, value = "después del impacto, la cadera retrocede {:.2f} torsos", -value
    return sentence.format(value).capitalize() + ".", pros


def body_measures(answer: dict) -> list[dict]:
    """The measures that separate amateurs from professionals next to the professionals'."""
    return [{"aspect": aspect, "label": BODY_TEXT[aspect][0], "mine": measured(aspect, answer)[0],
             "pros": measured(aspect, answer)[1],
             "in_range": answer["body"][aspect] <= float(row["pros_p90"])}
            for aspect, row in POINTS.items() if row["described"]]  # fmt: skip


def body_evidence(aspect: str, answer: dict) -> str:
    """How far a body point can be trusted, in Spanish: its level and its rates on our videos."""
    row = POINTS[aspect]
    if row["source"] == "measure":
        return ("Es una medida de la pose, sin entrenar: quedas fuera del rango de 9 de cada 10 "
                f"profesionales. De los saques amateur fuera de ese rango, {float(row['precision']):.0%} "
                "tenían este problema según quien los etiquetó. No es un diagnóstico.")  # fmt: skip
    mine = answer["needs_work"][aspect]
    cut = BODY[row["model"]]["thresholds"][aspect]
    return (f"{sum(s >= cut for s in mine['seeds'])} de {len(mine['seeds'])} versiones del modelo "
            f"lo marcan. En validación acertó {float(row['precision']):.0%} de las veces que lo "
            f"marcó y lo marcó en {float(row['pros_rate']):.0%} de los saques profesionales. Está "
            "en prueba: falta probarlo con jugadores nuevos.")  # fmt: skip


def points(answer: dict) -> list[dict]:
    """Up to MAX_POINTS points to improve, the surest level first and within a level the furthest
    past its threshold: the tossing arm (validated) and the aspects of POINTS with a level, by
    their body model or by their measure past the professionals' range. An aspect read from the
    legs is left out when MediaPipe did not see them, and the landing when the video ends before
    it."""
    found = []
    if answer["drops_early"]:
        found.append((RANK["validated"], answer["probability"] - answer["threshold"], {
            "aspect": "toss", "label": "Brazo que lanza", "level": "validated",
            "seen": observation(answer), "evidence": certainty(answer), "drill_id": "toss_hold"}))  # fmt: skip
    for aspect, row in POINTS.items():
        unseen = aspect in LEGS and (answer["legs_seen"] < MIN_SEEN
                                     or (aspect == "landing" and not answer["landing_seen"]))  # fmt: skip
        if not row["level"] or unseen:
            continue
        if row["source"] == "model":
            margin = (answer["needs_work"][aspect]["probability"]
                      - BODY[row["model"]]["thresholds"][aspect])  # fmt: skip
        else:
            margin = (answer["body"][aspect] - float(row["threshold"])) / float(row["scale"])
        if margin >= 0:
            found.append((RANK[row["level"]], margin, {
                "aspect": aspect, "label": BODY_TEXT[aspect][0], "level": row["level"],
                "seen": " ".join(measured(aspect, answer)),
                "evidence": body_evidence(aspect, answer), "drill_id": BODY_TEXT[aspect][2]}))  # fmt: skip
    found.sort(key=lambda point: point[:2], reverse=True)
    return [point | {"level_label": LEVELS[point["level"]]} for *_, point in found[:MAX_POINTS]]


def stretch_text(start: float, stop: float) -> str:
    if stop <= 0:
        return f"entre {-start:.2f} y {-stop:.2f} s antes del impacto"
    if start >= 0:
        return f"entre {start:.2f} y {stop:.2f} s después del impacto"
    return f"de {-start:.2f} s antes a {stop:.2f} s después del impacto"


def moments(keypoints: np.ndarray, fps: float, answer: dict) -> dict[str, int]:
    """Source frame of each key moment: the window start, the tossing wrist at its highest, the
    last frame with it above the shoulder (absent when never above) and contact."""
    contact = answer["contact"]
    start = max(contact - round(dataset.BEFORE / dataset.FPS * fps), 0)
    wrist = keypoints[
        start : contact + 1, pose.LANDMARKS.index(f"{answer['tossing_side']}_wrist"), 1
    ]
    frames = {"start": start}
    if not np.isnan(wrist).all():
        frames["arm_up"] = start + int(np.nanargmin(wrist))  # image y grows downward
    if answer["drop_seconds"] <= dataset.BEFORE / dataset.FPS:
        frames["arm_down"] = contact - round(answer["drop_seconds"] * fps)
    frames["contact"] = contact
    return frames


def drawn(frame: np.ndarray, keypoints: np.ndarray, side: str, box) -> np.ndarray:
    """The frame with the skeleton, the tossing arm thicker in orange, cropped to the player."""
    out = pose.draw(frame, keypoints)
    arm = [keypoints[pose.LANDMARKS.index(f"{side}_{j}")] for j in ("shoulder", "elbow", "wrist")]
    points = [tuple(int(v) for v in p[:2]) if p[3] >= 0.5 else None for p in arm]  # NaN: False
    thickness = max(4, round(frame.shape[0] / 150))
    for a, b in pairwise(points):
        if a and b:
            cv2.line(out, a, b, (0, 140, 255), thickness, cv2.LINE_AA)
    x0, y0, x1, y1 = box
    return resize(out[max(y0, 0) : y1, max(x0, 0) : x1], IMAGE_SIDE)


def render(path, poses: dict, answer: dict, out: Path):
    """Key moment images and the slow motion from the window start to 0.5 s after contact, each
    frame labeled with its time from contact and the latest key moment."""
    fps, keypoints, side = poses["fps"], poses["keypoints"], answer["tossing_side"]
    frames = moments(keypoints, fps, answer)
    first, last = frames["start"], answer["contact"] + round(dataset.AFTER / dataset.FPS * fps)
    images = {}

    def labeled():
        for index, frame in video.iter_frames(path):
            if index > last:
                break
            if index < first:
                continue
            image = drawn(frame, keypoints[index], side, poses["box"])
            for name, at in frames.items():
                if at == index:
                    images[name] = image
            latest = MOMENTS[max((at, n) for n, at in frames.items() if at <= index)[1]]
            yield label(image, f"{(index - answer['contact']) / fps:+.2f} s  {latest}", 0.045)

    video.write_video(out, labeled(), fps / CONFIG["app"]["slowdown"])
    return {n: images[n] for n in frames if n in images}


def run_analysis(video_path, progress) -> dict:
    """Everything that takes time. Returns {"retake": reason}, {"error": message} or the result."""
    start = time.perf_counter()
    try:
        progress(0.1, desc="Paso 1 de 4: revisando el video… (unos 10 s en total)")
        check_video(video.probe(video_path), CONFIG["analysis"]["max_seconds"])
        progress(0.2, desc="Paso 2 de 4: buscando tu cuerpo en cada cuadro…")
        poses = pose.extract(video_path, CONFIG["pose"]["model"])
    except InsufficientEvidence as why:
        return {"retake": str(why)}
    except (ValueError, FileNotFoundError):
        return {"error": "No se pudo leer el video. Sube un archivo .mp4 o .mov de un solo saque."}
    if np.isnan(poses["keypoints"][:, 0, 0]).mean() > 1 - MIN_FOUND:
        return {"retake": "No encontramos a una persona en la mayor parte del video. Graba de lado, "
                          "con todo tu cuerpo dentro del cuadro."}  # fmt: skip
    progress(0.7, desc="Paso 3 de 4: midiendo el brazo que lanza…")
    try:
        answer = model.predict(MODEL, poses["keypoints"], poses["fps"])
    except ValueError:
        return {"retake": "No vemos el lanzamiento de la pelota: graba el saque completo, con el "
                          "brazo que lanza y la raqueta arriba dentro del cuadro."}  # fmt: skip
    if answer["shoulders"] < model.MIN_SHOULDERS:
        return {"retake": "La cámara no te ve de lado: un hombro tapa al otro. Graba de lado, con la "
                          "cámara mirando tu pecho, sobre la prolongación de la línea de fondo."}  # fmt: skip
    if answer["seen"] < MIN_SEEN:
        return {"retake": f"Solo vimos el {answer['seen']:.0%} de los puntos de tus brazos cerca "
                          "del impacto. Graba de lado, con más luz y sin que nada tape el brazo."}  # fmt: skip
    x, mask = dataset.window(poses["keypoints"], poses["fps"], answer["contact"])
    values = body.measures(x[None], mask[None], audited=True)
    answer["toss_height"] = float(body.toss_height(x[None], mask[None])[0])
    answer["body"] = {aspect: float(values[aspect][0]) for aspect in BODY_TEXT}
    answer["needs_work"] = {}
    for aspect, row in POINTS.items():
        if row["source"] == "model":
            m3 = BODY[row["model"]]
            o, k = (
                body.oriented(x[None], mask[None], m3["at_reach"])[0],
                m3["aspects"].index(aspect),
            )
            seeds = [float(body.probabilities(n, o, m3)[0, k]) for n in m3["networks"]]
            answer["needs_work"][aspect] = {"probability": float(np.mean(seeds)),
                                            "seeds": np.round(seeds, 4).tolist()}  # fmt: skip
    answer["legs_seen"] = float(mask[:, body.LOWER].mean())  # share of leg points MediaPipe saw
    answer["landing_seen"] = bool(mask[dataset.BEFORE + 1 :].any(1).all())  # the video goes on
    progress(0.85, desc="Paso 4 de 4: preparando la cámara lenta…")
    slow = OUTPUT_DIR / f"slow-{time.time_ns()}.mp4"
    images = render(video_path, poses, answer, slow)
    return {"answer": answer, "images": images, "slow": str(slow),
            "seconds": time.perf_counter() - start}  # fmt: skip


def cache_folder(video_path) -> Path:
    """One folder per video content, model weights, pose model and ANSWER."""
    digest = hashlib.sha256(Path(video_path).read_bytes())
    digest.update(WEIGHTS + BODY_WEIGHTS)
    digest.update(f"|{MODEL['threshold']}|{CONFIG['pose']['model']}|{ANSWER}".encode())
    return CACHE_DIR / digest.hexdigest()[:16]


def analysis(video_path, progress) -> dict:
    """run_analysis; for a demo video the answer is saved once and then shown at once."""
    folder = cache_folder(video_path) if DEMO_FOLDERS else None
    if folder not in DEMO_FOLDERS:
        return run_analysis(video_path, progress)
    if (folder / "result.json").exists():
        result = json.loads((folder / "result.json").read_text())
        if "answer" in result:
            result["images"] = {n: cv2.imread(str(folder / f"{n}.jpg")) for n in result["moments"]}
            result["slow"] = str(folder / "slow.mp4")
        return {**result, "cached": True}

    result = run_analysis(video_path, progress)
    if "error" in result:  # failures are not saved: the next try may work
        return result
    folder.mkdir(parents=True, exist_ok=True)
    saved = {k: v for k, v in result.items() if k not in ("images", "slow")}
    if "answer" in result:
        for name, image in result["images"].items():
            cv2.imwrite(str(folder / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copy(result["slow"], folder / "slow.mp4")
        saved["moments"] = list(result["images"])
    (folder / "result.json").write_text(json.dumps(saved, ensure_ascii=False))  # written last
    return result


def prepare_demo():
    """Analyze each demo video once (or load its saved answer) before the app opens."""
    for n, path in enumerate(DEMO_VIDEOS, 1):
        DEMO_FOLDERS.add(cache_folder(path))
        result = analysis(path, lambda *args, **kwargs: None)
        answer = result.get("answer")
        text = (f"P = {answer['probability']:.2f}" if answer
                else result.get("retake") or result["error"])  # fmt: skip
        print(f"Demo {n}/{len(DEMO_VIDEOS)}: {path.name}: {text}", flush=True)
