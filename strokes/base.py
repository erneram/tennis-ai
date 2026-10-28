"""Interface every stroke implements, plus the two model calls shared by all strokes.

A stroke is a package strokes/<name>/ whose __init__.py exposes the Stroke attributes below,
with criteria.yaml, drills.yaml and prompts/<version>.yaml next to it. Adding a stroke means
adding a folder with the same structure; nothing in core/ changes.

Only the language model looks at the video:
1. locate_phases: from evenly spaced frames, it decides whether the video can be analyzed and
   picks the frame that best shows each phase.
2. feedback: from those key frames, it chooses the main improvement and a drill.
"""

from importlib import import_module
from pathlib import Path
from string import Template
from typing import Protocol

import numpy as np
import yaml

from core.analysis import InsufficientEvidence, label, resize

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


def phases_schema(stroke: Stroke, n_frames: int, reasons: list[str]) -> dict:
    """JSON schema of locate_phases: "none" or the reason to record again, then the number
    (1 to n_frames) of the frame that best shows each phase."""
    frame = {"type": "integer", "enum": list(range(1, n_frames + 1))}
    return {
        "type": "object",
        "properties": {
            "retake": {"type": "string", "enum": ["none", *reasons]},
            **{phase: frame for phase in stroke.PHASES},
        },
        "required": ["retake", *stroke.PHASES],
        "additionalProperties": False,
    }


def locate_phases(stroke: Stroke, images: list[np.ndarray], vlm, version: str, **generation):
    """First model call. Returns (phase -> index into `images`, Generation).

    Raises InsufficientEvidence with the message for the player when the model asks for a new
    recording. Every frame is numbered on the image, so the model can refer to it.
    """
    prompts = load(stroke, f"prompts/{version}.yaml")["phases"]
    criteria = load(stroke, "criteria.yaml")
    retake = prompts["retake"]
    user = Template(prompts["user"]).substitute(
        n_frames=len(images),
        phase_guide="\n".join(f"- {p}: {s['looks_like']}" for p, s in criteria["phases"].items()),
        reasons="\n".join(f"- {reason}: {spec['when']}" for reason, spec in retake.items()),
    )
    numbered = [
        label(resize(image, LOCATE_SIDE), str(n), height=0.1) for n, image in enumerate(images, 1)
    ]
    schema = phases_schema(stroke, len(images), list(retake))
    result = vlm.generate_json(prompts["system"], user, numbered, schema, **generation)
    if result.data["retake"] != "none":
        raise InsufficientEvidence(retake[result.data["retake"]]["message"])

    key_frames, last = {}, 0
    for phase in stroke.PHASES:
        # ponytail: a frame that goes back in time is moved up to the previous phase's frame
        last = max(last, result.data[phase] - 1)
        key_frames[phase] = last
    return key_frames, result


def feedback_schema(stroke: Stroke) -> dict:
    """JSON schema of one feedback answer. The model can only pick known phases, categories and
    drills."""
    criteria, drills = load(stroke, "criteria.yaml"), load(stroke, "drills.yaml")["drills"]
    phases = list(stroke.PHASES)
    return {
        "type": "object",
        "properties": {
            "phase": {"type": "string", "enum": phases},
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
