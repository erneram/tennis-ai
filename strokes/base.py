"""Interface every stroke implements, plus the two model calls shared by all strokes.

A stroke is a package strokes/<name>/ whose __init__.py exposes the Stroke attributes below,
with criteria.yaml, drills.yaml and prompts/<version>.yaml next to it. Adding a stroke means
adding a folder with the same structure; nothing in core/ changes.

Only the language model looks at the video:
1. find_window (prompt v5 and later): from evenly spaced frames of the whole video, it decides
   whether the video can be analyzed and where the stroke itself starts and ends. Players
   bounce the ball before serving, so most of the video is not the stroke.
2. locate_phases: in frames sampled again inside that window, it picks the frame that best shows
   each phase. Up to v4 it worked on the whole video and also decided whether to record again.
3. feedback: from those key frames, the main improvement (or no_clear_problem) and a drill.
   From v6 on, one yes / no / unclear question per category, each about one key frame; from v7
   on, a choice between the good and the bad state instead of yes / no; from v8 on, the
   question about the good position and the one about the problem, kept only when they agree;
   from v11 on, some categories one neutral question with options instead, some on the upper
   body only (upper_body).
phase_frames runs 1 and 2; from v8 on, before them, find_player crops every frame around the
player; from v9 on, after them, refine_phases takes a closer look at the fastest phases; from
v13 on, offset_phases places some key frames by code, a fixed time from another one.
"""

from dataclasses import replace
from importlib import import_module
from itertools import pairwise
from pathlib import Path
from string import Template
from typing import Protocol

import cv2
import numpy as np
import yaml

from core.analysis import (
    FRAME_SIDE,
    Frames,
    InsufficientEvidence,
    crop,
    label,
    resample,
    resize,
    zoom,
)

STROKES_DIR = Path(__file__).resolve().parent
LOCATE_SIDE = 512  # frames for locate_phases: about 150 tokens each with Qwen3-VL


class Stroke(Protocol):
    NAME: str
    PHASES: tuple[str, ...]  # in chronological order
    LABELS: dict[str, str]  # phase -> name shown to the player


def get(name: str) -> Stroke:
    return import_module(f"strokes.{name}")


def load(stroke: Stroke, file: str) -> dict:
    """Read a versioned configuration file of the stroke: criteria.yaml, drills.yaml, prompts/v4.yaml."""
    return yaml.safe_load((STROKES_DIR / stroke.NAME / file).read_text())


def _numbered(images: list[np.ndarray]) -> list[np.ndarray]:
    """Every frame with its number written on it, so the model can refer to it."""
    return [
        label(resize(image, LOCATE_SIDE), str(n), height=0.1) for n, image in enumerate(images, 1)
    ]


def _retake_schema(reasons: list[str]) -> dict:
    return {"retake": {"type": "string", "enum": ["none", *reasons]}} if reasons else {}


def _ask_retake(retake: dict, data: dict) -> None:
    if data.get("retake", "none") != "none":
        raise InsufficientEvidence(retake[data["retake"]]["message"])


def window_schema(n_frames: int, reasons: list[str]) -> dict:
    """JSON schema of find_window: "none" or the reason to record again, then the numbers of the
    frames just before and just after the stroke."""
    frame = {"type": "integer", "enum": list(range(1, n_frames + 1))}
    retake = _retake_schema(reasons)
    return {
        "type": "object",
        "properties": {**retake, "start": frame, "end": frame},
        "required": [*retake, "start", "end"],
        "additionalProperties": False,
    }


def phases_schema(stroke: Stroke, n_frames: int, reasons: list[str] = ()) -> dict:
    """JSON schema of locate_phases: the number (1 to n_frames) of the frame that best shows each
    phase, after "retake" when this call also decides whether to record again (v4)."""
    frame = {"type": "integer", "enum": list(range(1, n_frames + 1))}
    retake = _retake_schema(reasons)
    return {
        "type": "object",
        "properties": {**retake, **{phase: frame for phase in stroke.PHASES}},
        "required": [*retake, *stroke.PHASES],
        "additionalProperties": False,
    }


def find_window(stroke: Stroke, images: list[np.ndarray], vlm, version: str, **generation):
    """Model call 1 (v5 and later). Returns (first, last, Generation): indices into `images` that
    contain the whole stroke, with one frame of margin on each side.

    Raises InsufficientEvidence with the message for the player when the model asks for a new
    recording.
    """
    prompts = load(stroke, f"prompts/{version}.yaml")["window"]
    retake = prompts["retake"]
    user = Template(prompts["user"]).substitute(
        n_frames=len(images),
        reasons="\n".join(f"- {reason}: {spec['when']}" for reason, spec in retake.items()),
    )
    schema = window_schema(len(images), list(retake))
    result = vlm.generate_json(prompts["system"], user, _numbered(images), schema, **generation)
    _ask_retake(retake, result.data)
    start, end = sorted((result.data["start"] - 1, result.data["end"] - 1))
    return max(start - 1, 0), min(end + 1, len(images) - 1), result


def locate_phases(stroke: Stroke, images: list[np.ndarray], vlm, version: str, **generation):
    """Model call 2. Returns (phase -> index into `images`, Generation).

    Up to v4 this was the first call and could raise InsufficientEvidence too.
    """
    prompts = load(stroke, f"prompts/{version}.yaml")["phases"]
    criteria = load(stroke, "criteria.yaml")
    retake = prompts.get("retake", {})
    user = Template(prompts["user"]).substitute(
        n_frames=len(images),
        phase_guide="\n".join(f"- {p}: {s['looks_like']}" for p, s in criteria["phases"].items()),
        reasons="\n".join(f"- {reason}: {spec['when']}" for reason, spec in retake.items()),
    )
    schema = phases_schema(stroke, len(images), list(retake))
    result = vlm.generate_json(prompts["system"], user, _numbered(images), schema, **generation)
    _ask_retake(retake, result.data)

    key_frames, last = {}, 0
    for phase in stroke.PHASES:
        # ponytail: a frame that goes back in time is moved up to the previous phase's frame
        last = max(last, result.data[phase] - 1)
        key_frames[phase] = last
    return key_frames, result


def player_schema() -> dict:
    """JSON schema of find_player: one box in Qwen-VL's own format, 0 to 1000 on each side."""
    coordinate = {"type": "integer", "minimum": 0, "maximum": 1000}
    box = {"type": "array", "items": coordinate, "minItems": 4, "maxItems": 4}
    return {
        "type": "object",
        "properties": {"bbox_2d": box},
        "required": ["bbox_2d"],
        "additionalProperties": False,
    }


def upper_body(image: np.ndarray, vlm, prompts: dict, **generation):
    """Model call (v11): the player's box on `image` (the `player` prompt), cut to the head,
    shoulders and arms and enlarged to FRAME_SIDE. Returns (image, Generation); the image
    unchanged when the box is empty.

    On v10's leg-drive frames, cropped to the player's reach, the model answered "above the
    head" about the tossing hand on all 32 videos; on the upper body only, it saw the arm at
    shoulder height on 3 of the 7 JC frames that show it (experiments/probe_toss_arm.py).
    """
    result = vlm.generate_json(prompts["system"], prompts["user"], [image], player_schema(),
                               **generation)  # fmt: skip
    x0, y0, x1, y1 = (v / 1000 for v in result.data["bbox_2d"])
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return image, result
    part = crop(image, (max(0, x0 - 0.15 * w), max(0, y0 - 0.15 * h), min(1, x1 + 0.15 * w),
                        min(1, y0 + 0.6 * h)))  # fmt: skip
    scale = FRAME_SIDE / max(part.shape[:2])
    return cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), result


def reach(bbox: list[int], reach: dict, width: int, height: int):
    """The player's box (0-1000) grown by `reach` (shares of the player's height: up, down, and
    to each side) so the whole stroke fits, as (x0, y0, x1, y1) shares of the frame. None when
    the box is empty or too small to be a person."""
    x1, y1, x2, y2 = (v / 1000 for v in bbox)
    tall = (y2 - y1) * height  # pixels
    if x2 <= x1 or tall < 0.05 * height:
        return None
    side = reach["side"] * tall / width  # as a share of the width
    return (max(x1 - side, 0.0), max(y1 - reach["up"] * tall / height, 0.0),
            min(x2 + side, 1.0), min(y2 + reach["down"] * tall / height, 1.0))  # fmt: skip


def find_player(stroke: Stroke, images: list[np.ndarray], vlm, version: str, **generation):
    """Model calls (v8 on), on the first, middle and last frames of the video: the box around
    the player who serves in each, joined and grown to the stroke's reach. Returns (box or None,
    [Generation]).

    With the player far from the camera, the phase frames were six frames in a row on 8 of 20
    videos; cropped by hand, on 1 of 20. The box is None when it keeps more than `max_share` of
    the frame: such a crop gains nothing, and even a small change of the frames can turn the
    phases into a guess. A wrong box on one frame makes the joined box large, so it is dropped.
    """
    prompts = load(stroke, f"prompts/{version}.yaml")["player"]
    results = [
        vlm.generate_json(
            prompts["system"], prompts["user"], [image], player_schema(), **generation
        )
        for image in (images[0], images[len(images) // 2], images[-1])
    ]
    boxes = [r.data["bbox_2d"] for r in results]
    joined = [
        *(min(b[i] for b in boxes) for i in (0, 1)),
        *(max(b[i] for b in boxes) for i in (2, 3)),
    ]
    height, width = images[0].shape[:2]
    box = reach(joined, prompts["reach"], width, height)
    if box is not None and (box[2] - box[0]) * (box[3] - box[1]) > prompts["max_share"]:
        box = None
    return box, results


def phase_frames(stroke: Stroke, frames: Frames, vlm, version: str, **generation):
    """Model calls 1 and 2. Returns (frames, phase -> index into frames.images, [Generation]).

    From v5 on, the phases are located in frames sampled again inside the stroke's window, and
    those are the frames returned. Raises InsufficientEvidence, or ValueError when the phase
    frames are six frames in a row: the stroke fills the window, so the model counted instead of
    looking.
    """
    prompts = load(stroke, f"prompts/{version}.yaml")
    calls = []
    if "player" in prompts:  # before the window: on the full frame, the window was wrong
        box, results = find_player(stroke, frames.images, vlm, version, **generation)
        calls += results
        if box is not None:
            frames = zoom(frames, box)
    if "window" in prompts:  # resample keeps the crop
        first, last, result = find_window(stroke, frames.images, vlm, version, **generation)
        calls.append(result)
        frames = resample(frames, frames.indices[first], frames.indices[last])
    key_frames, result = locate_phases(stroke, frames.images, vlm, version, **generation)
    calls.append(result)
    step = prompts["phases"].get("retry_step")  # v9: a second try on every step-th frame
    if step and "window" in prompts and _guess(key_frames):
        # with the seed fixed, the same frames give the same guess: the player could never
        # get an answer, so the second try sees other frames
        fewer, result = locate_phases(stroke, frames.images[::step], vlm, version, **generation)
        calls.append(result)
        if not _guess(fewer):
            key_frames = {phase: i * step for phase, i in fewer.items()}
    if "window" in prompts and _guess(key_frames):
        raise ValueError(f"Phase frames look like a guess: {list(key_frames.values())}")
    if "refine" in prompts:
        frames, key_frames, results = refine_phases(stroke, frames, key_frames, vlm, version,
                                                    **generation)  # fmt: skip
        calls += results
    if "offsets" in prompts:
        frames, key_frames = offset_phases(stroke, frames, key_frames, prompts["offsets"])
    return frames, key_frames, calls


def _guess(key_frames: dict[str, int]) -> bool:
    """Six frames in a row: the model counted instead of looking."""
    return all(b - a == 1 for a, b in pairwise(key_frames.values()))


def frame_schema(n_frames: int) -> dict:
    """JSON schema of refine_phases: the number of one frame."""
    frame = {"type": "integer", "enum": list(range(1, n_frames + 1))}
    return {
        "type": "object",
        "properties": {"frame": frame},
        "required": ["frame"],
        "additionalProperties": False,
    }


def refine_phases(stroke: Stroke, frames: Frames, key_frames: dict[str, int], vlm,
                  version: str, **generation):  # fmt: skip
    """Model calls 2b (v9 on): for each phase in `refine`, frames sampled closely between the
    key frames of `from` and `to` (null: the end of the window), and the frame that best
    matches its description. Returns (frames with the chosen frames added in time order,
    phase -> index into them, [Generation]).

    The contact and the landing last a few hundredths of a second: among 32 frames over the
    whole serve, the model took the racket behind the head for the contact (6 of 18 JC
    videos) and a frame still in the air for the follow-through (5 of 18).
    """
    prompts = load(stroke, f"prompts/{version}.yaml")["refine"]
    calls = []
    for phase, spec in prompts["phases"].items():
        first = frames.indices[key_frames[spec["from"]]]
        last = frames.indices[key_frames[spec["to"]] if spec["to"] else -1]
        if last - first < 2:  # nothing closer to look at
            continue
        close = resample(frames, first, last, prompts["frames"])
        user = Template(prompts["user"]).substitute(
            n_frames=len(close.images), looks_like=spec["looks_like"]
        )
        result = vlm.generate_json(prompts["system"], user, _numbered(close.images),
                                   frame_schema(len(close.images)), **generation)  # fmt: skip
        calls.append(result)
        pick = result.data["frame"] - 1
        frames, key_frames = _add_frame(frames, key_frames, phase, close.indices[pick],
                                        close.images[pick])  # fmt: skip
    return frames, key_frames, calls


def offset_phases(stroke: Stroke, frames: Frames, key_frames: dict[str, int],
                  offsets: dict) -> tuple[Frames, dict[str, int]]:  # fmt: skip
    """v13: each phase in `offsets` is the source frame `seconds` from the key frame of `from`,
    kept between the key frames of the phases before and after it. Returns (frames with it
    added, phase -> index into them), as refine_phases.

    Among 12 close frames the model put the leg-drive frame within 0.1 s of contact on 7 of 21
    JC serves (v10, v12) and 0.4 to 1.2 s before it on 18 with another description (v11); the
    contact frame it finds well (16 of 18, v9).
    """
    for phase, spec in offsets.items():
        position = stroke.PHASES.index(phase)
        before = frames.indices[key_frames[stroke.PHASES[position - 1]]]
        after = frames.indices[key_frames[stroke.PHASES[position + 1]]]
        if after - before < 2:  # no frame between them
            continue
        index = frames.indices[key_frames[spec["from"]]] + round(spec["seconds"] * frames.fps)
        index = min(max(index, before + 1), after - 1)
        [image] = resample(frames, index, index, 1).images
        frames, key_frames = _add_frame(frames, key_frames, phase, index, image)
    return frames, key_frames


def _add_frame(frames: Frames, key_frames: dict[str, int], phase: str, index: int,
               image: np.ndarray) -> tuple[Frames, dict[str, int]]:  # fmt: skip
    """`frames` with source frame `index` added in time order as the key frame of `phase`."""
    sources = {p: frames.indices[i] for p, i in key_frames.items()}
    sources[phase] = index
    images = dict(zip(frames.indices, frames.images, strict=True))
    images[index] = image
    order = sorted(images)
    frames = replace(frames, indices=order, images=[images[i] for i in order])
    return frames, {p: order.index(i) for p, i in sources.items()}


def feedback_schema(stroke: Stroke) -> dict:
    """JSON schema of one feedback answer. The model can only pick known phases, categories and
    drills. "seen" comes first: the model writes what the frames show before it judges them.
    Phase "none" goes with no_clear_problem."""
    criteria, drills = load(stroke, "criteria.yaml"), load(stroke, "drills.yaml")["drills"]
    phases = list(stroke.PHASES)
    return {
        "type": "object",
        "properties": {
            "seen": {"type": "string", "maxLength": 600},
            "phase": {"type": "string", "enum": [*phases, "none"]},
            "category": {"type": "string", "enum": list(criteria["categories"])},
            "observation": {"type": "string", "maxLength": 400},
            "evidence_phases": {
                "type": "array",
                "items": {"type": "string", "enum": phases},
                "minItems": 1,
                "maxItems": 3,
            },
            "drill_id": {"type": "string", "enum": [d["id"] for d in drills]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": [
            "seen",
            "phase",
            "category",
            "observation",
            "evidence_phases",
            "drill_id",
            "confidence",
        ],
        "additionalProperties": False,
    }


def check_schema(answers: tuple[str, ...] = ("yes", "no", "unclear")) -> dict:
    """JSON schema of one check: what the frame shows, then yes / no / unclear (v6) or the
    good state / the bad state / unclear (v7)."""
    return {
        "type": "object",
        "properties": {
            "seen": {"type": "string", "maxLength": 200},
            "answer": {"type": "string", "enum": list(answers)},
        },
        "required": ["seen", "answer"],
        "additionalProperties": False,
    }


def _usage(calls: list) -> dict:
    return {
        "prompt_tokens": sum(c.prompt_tokens for c in calls),
        "generation_tokens": sum(c.generation_tokens for c in calls),
        "llm_seconds": round(sum(c.seconds for c in calls), 2),
        "peak_memory_gb": round(max(c.peak_memory_gb for c in calls), 2),
    }


def feedback(stroke: Stroke, images: dict[str, np.ndarray], vlm, variant: str, version: str,
             **generation) -> dict:  # fmt: skip
    """The main improvement, judged from the key frame of each phase.

    variant "full" sends the frames + what to observe + criteria + drills; "simple" (baseline)
    sends the frames with only the category and drill names. From v6 on, "full" asks one
    question per category instead (checked_feedback).
    """
    prompts = load(stroke, f"prompts/{version}.yaml")
    if variant == "full" and "checks" in prompts:
        return checked_feedback(stroke, images, vlm, prompts, version, **generation)
    prompts = prompts["feedback" if variant == "full" else variant]
    criteria, drills = load(stroke, "criteria.yaml"), load(stroke, "drills.yaml")["drills"]
    categories = criteria["categories"]
    fields = {
        "frame_list": ", ".join(f"{i}. {p}" for i, p in enumerate(stroke.PHASES, 1)),
        "phase_guide": "\n".join(f"- {p}: {s['observe']}" for p, s in criteria["phases"].items()),
        "categories": "\n".join(
            f"- {c['phase']}: {k}: {c['description']}" for k, c in categories.items()
        ),
        "drills": "\n".join(f"- {d['id']}: {d['category']}: {d['name']}" for d in drills),
        "category_names": ", ".join(categories),
        "drill_names": ", ".join(f"{d['id']} ({d['category']})" for d in drills),
    }
    user = Template(prompts["user"]).substitute(fields)
    frames = [images[p] for p in stroke.PHASES]
    result = vlm.generate_json(
        prompts["system"], user, frames, feedback_schema(stroke), **generation
    )
    data = result.data
    return {
        **data,
        "consistent": categories[data["category"]]["phase"] == data["phase"],
        "source": f"vlm:{variant}:{version}",
        **_usage([result]),
    }


def checked_feedback(stroke: Stroke, images: dict[str, np.ndarray], vlm, prompts: dict,
                     version: str, **generation) -> dict:  # fmt: skip
    """v6: one yes / no / unclear question per category (its `check` in criteria.yaml), each
    about the key frame of its phase only. A single choice among all categories followed the
    order of the list, not the frames (v5).

    Each question asks about the good position, so "no" is the problem. The main improvement
    is the "no" of the earliest phase, as in the labeling guide; one more call writes the
    observation for it. With no "no" the answer is no_clear_problem, with confidence "low" if
    any check was unclear. Phase, category and drill are chosen by code, so the answer is
    always consistent.

    v7 (prompts["checks"]["choose"]): the model picks the good or the bad state of the body
    part (`choose` in criteria.yaml) instead of answering yes / no; the good state is stored as
    "yes" and the bad one as "no".

    v8 (prompts["checks"]["ask"]): several questions per category, each a criteria field and
    what its "yes" means: `check` (good position) and `problem`. v6 always found the position
    good and v7 always found the problem, so the wording decided. A check is "yes" or "no" only
    when every question agrees, otherwise "unclear".

    v10 (prompts["checks"]["questions"]): the version's own wording of some categories' fields,
    over criteria.yaml, so the earlier versions still ask what they asked. checks.frames: the
    phase whose key frame a category is judged on, when not its own (stored in each check).

    v11: a category with `options` (answer -> yes / no / unclear) is asked one neutral question
    instead (prompts["checks"]["options"]): with yes / no the model agreed with what the
    question described. `upper_body`: judged on the upper body only (upper_body()).
    """
    criteria, drills = load(stroke, "criteria.yaml"), load(stroke, "drills.yaml")["drills"]
    phases, categories = criteria["phases"], criteria["categories"]
    choose = prompts["checks"].get("choose", False)
    ask = prompts["checks"].get("ask", {"check": "good"})
    questions = prompts["checks"].get("questions", {})
    frames = prompts["checks"].get("frames", {})
    calls, checks, seen_on = [], {}, {}
    for category, spec in categories.items():
        if "check" not in spec:  # no_clear_problem
            continue
        spec = {**spec, **questions.get(category, {})}
        phase = frames.get(category, spec["phase"])  # the key frame it is judged on
        image = images[phase]
        if spec.get("upper_body"):
            image, result = upper_body(image, vlm, prompts["player"], **generation)
            calls.append(result)
        seen_on[category] = image
        states = spec["choose"] if choose else {"part": "", "good": "yes", "bad": "no"}
        votes, seen = {}, None
        if "options" in spec:
            options = [*spec["options"], "unclear"]
            user = Template(prompts["checks"]["options"]["user"]).substitute(
                question=spec["question"], options="; ".join(options)
            )
            result = vlm.generate_json(prompts["checks"]["options"]["system"], user, [image],
                                       check_schema(tuple(options)), **generation)  # fmt: skip
            calls.append(result)
            votes["question"] = spec["options"].get(result.data["answer"], "unclear")
            seen = result.data["seen"]
        else:
            for field, yes_means in ask.items():
                user = Template(prompts["checks"]["user"]).substitute(
                    phase=phase, looks_like=phases[phase]["looks_like"], question=spec[field],
                    **states,
                )  # fmt: skip
                schema = check_schema((states["good"], states["bad"], "unclear"))
                result = vlm.generate_json(prompts["checks"]["system"], user, [image],
                                           schema, **generation)  # fmt: skip
                calls.append(result)
                vote = {states["good"]: "yes", states["bad"]: "no"}.get(result.data["answer"])
                if yes_means == "bad":  # a "yes" to the problem question is the problem
                    vote = {"yes": "no", "no": "yes"}.get(vote)
                votes[field], seen = vote or "unclear", seen or result.data["seen"]
        answer = next(iter(votes.values())) if len(set(votes.values())) == 1 else "unclear"
        checks[category] = {"seen": seen, "answer": answer, "votes": votes, "phase": phase}

    found = [c for c, check in checks.items() if check["answer"] == "no"]
    if found:
        category = min(found, key=lambda c: stroke.PHASES.index(categories[c]["phase"]))
        phase, frame = categories[category]["phase"], checks[category]["phase"]
        user = Template(prompts["observation"]["user"]).substitute(
            phase=phase,
            description=categories[category]["description"],
            observe=phases[phase]["observe"],
            seen=checks[category]["seen"],
        )
        schema = {
            "type": "object",
            "properties": {"observation": {"type": "string", "maxLength": 400}},
            "required": ["observation"],
            "additionalProperties": False,
        }
        result = vlm.generate_json(prompts["observation"]["system"], user, [seen_on[category]],
                                   schema, **generation)  # fmt: skip
        calls.append(result)
        observation, confidence, evidence = result.data["observation"], "medium", [frame]
    else:
        category, phase = "no_clear_problem", "none"
        observation = prompts["observation"]["no_clear_problem"].strip()
        unclear = any(check["answer"] == "unclear" for check in checks.values())
        confidence, evidence = ("low" if unclear else "high"), []
    return {
        "phase": phase,
        "category": category,
        "observation": observation,
        "evidence_phases": evidence,
        "drill_id": next(d["id"] for d in drills if d["category"] == category),
        "confidence": confidence,
        "checks": checks,
        "consistent": True,
        "source": f"vlm:full:{version}",
        **_usage(calls),
    }
