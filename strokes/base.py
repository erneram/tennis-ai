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
3. feedback: from those key frames, it describes what it sees, then chooses the main improvement
   (or no_clear_problem) and a drill.
phase_frames runs 1 and 2.
"""

from importlib import import_module
from itertools import pairwise
from pathlib import Path
from string import Template
from typing import Protocol

import numpy as np
import yaml

from core.analysis import Frames, InsufficientEvidence, label, resample, resize

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


def phase_frames(stroke: Stroke, frames: Frames, vlm, version: str, **generation):
    """Model calls 1 and 2. Returns (frames, phase -> index into frames.images, [Generation]).

    From v5 on, the phases are located in frames sampled again inside the stroke's window, and
    those are the frames returned. Raises InsufficientEvidence, or ValueError when the phase
    frames are six frames in a row: the stroke fills the window, so the model counted instead of
    looking.
    """
    calls = []
    if "window" in load(stroke, f"prompts/{version}.yaml"):
        first, last, result = find_window(stroke, frames.images, vlm, version, **generation)
        calls.append(result)
        frames = resample(frames, frames.indices[first], frames.indices[last])
    key_frames, result = locate_phases(stroke, frames.images, vlm, version, **generation)
    calls.append(result)
    if len(calls) == 2 and all(b - a == 1 for a, b in pairwise(key_frames.values())):
        raise ValueError(f"Phase frames look like a guess: {list(key_frames.values())}")
    return frames, key_frames, calls


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


def feedback(stroke: Stroke, images: dict[str, np.ndarray], vlm, variant: str, version: str,
             **generation) -> dict:  # fmt: skip
    """Second model call: the main improvement, judged from the key frame of each phase.

    variant "full" sends the frames + what to observe + criteria + drills; "simple" (baseline)
    sends the frames with only the category and drill names.
    """
    prompts = load(stroke, f"prompts/{version}.yaml")["feedback" if variant == "full" else variant]
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
        "prompt_tokens": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
        "llm_seconds": round(result.seconds, 2),
        "peak_memory_gb": round(result.peak_memory_gb, 2),
    }
