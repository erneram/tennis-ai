"""Interface every stroke implements, plus helpers shared by all strokes.

A stroke is a package strokes/<name>/ whose __init__.py exposes the Stroke attributes below,
with criteria.yaml, drills.yaml and prompts/<version>.yaml next to it. Adding a stroke means
adding a folder with the same structure; nothing in core/ changes.
"""

from importlib import import_module
from pathlib import Path
from string import Template
from typing import Protocol

import numpy as np
import yaml

STROKES_DIR = Path(__file__).resolve().parent


class Stroke(Protocol):
    NAME: str
    PHASES: tuple[str, ...]  # in chronological order
    LABELS: dict[str, str]  # phase -> name shown to the player
    MEASUREMENT_LABELS: dict[str, str]  # measurement -> name shown to the player

    def segment(self, xy: np.ndarray, fps: float) -> dict[str, int]:
        """Phase -> first frame. Raises core.analysis.InsufficientEvidence if phases are not found."""

    def key_frames(self, phases: dict[str, int], n_frames: int, fps: float) -> dict[str, int]:
        """Phase -> frame that best shows it (sent to the model and shown to the player)."""

    def measure(self, xy: np.ndarray, fps: float, phases: dict[str, int]) -> dict[str, float]:
        """Measurement name -> value, computed only from the pose."""


def get(name: str) -> Stroke:
    return import_module(f"strokes.{name}")


def load(stroke: Stroke, file: str) -> dict:
    """Read a versioned configuration file of the stroke: criteria.yaml, drills.yaml, prompts/v1.yaml."""
    return yaml.safe_load((STROKES_DIR / stroke.NAME / file).read_text())


def check_results(criteria: dict, measurements: dict[str, float]) -> list[dict]:
    """Evaluate every measurable check of the coach criteria.

    `deviation` > 0 means the value is outside the coach range, relative to the limit.
    """
    results = []
    for phase, spec in criteria["phases"].items():
        for check in spec.get("checks", []):
            value = measurements.get(check["metric"])
            if value is None or np.isnan(value):
                continue
            deviation = -np.inf
            if "min" in check:
                deviation = (check["min"] - value) / max(abs(check["min"]), 1)
            if "max" in check:
                deviation = max(deviation, (value - check["max"]) / max(abs(check["max"]), 1))
            results.append({**check, "phase": phase, "value": value, "deviation": float(deviation)})
    return results


def rule_feedback(stroke: Stroke, measurements: dict[str, float]) -> dict:
    """Baseline without a language model: the check furthest outside (or closest to) its range wins."""
    criteria, drills = load(stroke, "criteria.yaml"), load(stroke, "drills.yaml")["drills"]
    results = check_results(criteria, measurements)
    if not results:
        raise ValueError("No measurable check: cannot give rule-based feedback")
    worst = max(results, key=lambda r: r["deviation"])
    category = criteria["categories"][worst["category"]]
    drill = next(d for d in drills if d["category"] == worst["category"])
    return {
        "phase": worst["phase"],
        "category": worst["category"],
        "observation": f"{category['description']} (medida {worst['metric']} = {worst['value']}).",
        "evidence_phases": [worst["phase"]],
        "drill_id": drill["id"],
        "confidence": "medium" if worst["deviation"] > 0 else "low",
        "source": "rules",
    }


def evidence_categories(criteria: dict, measurements: dict[str, float]) -> list[str]:
    """Categories the measurements leave open: the ones out of range (furthest first) or, if none
    is, the ones that could not be measured. An in-range measurement rules its category out."""
    results = check_results(criteria, measurements)
    out = [
        r["category"] for r in sorted(results, key=lambda r: -r["deviation"]) if r["deviation"] > 0
    ]
    if out:
        return list(dict.fromkeys(out))
    measured = {r["category"] for r in results}
    return [c for c in criteria["categories"] if c not in measured]


def feedback_schema(stroke: Stroke, allowed: list[str] | None = None) -> dict:
    """JSON schema of one feedback answer. The model can only pick known phases, categories and
    drills, restricted to the `allowed` categories when given."""
    criteria, drills = load(stroke, "criteria.yaml"), load(stroke, "drills.yaml")["drills"]
    categories = allowed or list(criteria["categories"])
    allowed_phases = {criteria["categories"][c]["phase"] for c in categories}
    phases = list(stroke.PHASES)
    return {
        "type": "object",
        "properties": {
            "phase": {"type": "string", "enum": [p for p in phases if p in allowed_phases]},
            "category": {"type": "string", "enum": categories},
            "observation": {"type": "string", "maxLength": 400},
            "evidence_phases": {
                "type": "array",
                "items": {"type": "string", "enum": phases},
                "minItems": 1,
                "maxItems": 3,
            },
            "drill_id": {
                "type": "string",
                "enum": [d["id"] for d in drills if d["category"] in categories],
            },
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


def _measurement_lines(criteria: dict, measurements: dict[str, float]) -> str:
    checks = {r["metric"]: r for r in check_results(criteria, measurements)}
    lines = []
    for name, value in measurements.items():
        check = checks.get(name)
        if check is None:
            lines.append(f"- {name} = {value} (informative)")
            continue
        limits = " and ".join(f"{k} {check[k]}" for k in ("min", "max") if k in check)
        status = "OUT OF RANGE" if check["deviation"] > 0 else "within range"
        lines.append(f"- {check['phase']}: {name} = {value} (coach range: {limits}) -> {status}")
    return "\n".join(lines)


def feedback(stroke: Stroke, analysis, vlm, variant: str = "full", version: str = "v1",
             **generation) -> dict:  # fmt: skip
    """Ask the vision-language model for the main improvement of one analyzed attempt.

    variant "full" sends frames + measurements + coach criteria; "simple" (baseline) sends frames only.
    A prompt with `constrain_to_evidence: true` limits the answer to evidence_categories().
    """
    prompts = load(stroke, f"prompts/{version}.yaml")["feedback" if variant == "full" else variant]
    criteria, drills = load(stroke, "criteria.yaml"), load(stroke, "drills.yaml")["drills"]
    categories = criteria["categories"]
    fields = {
        "frame_list": ", ".join(f"{i}. {p}" for i, p in enumerate(stroke.PHASES, 1)),
        "phase_guide": "\n".join(f"- {p}: {s['observe']}" for p, s in criteria["phases"].items()),
        "measurements": _measurement_lines(criteria, analysis.measurements),
        "categories": "\n".join(
            f"- {c['phase']}: {k}: {c['description']}" for k, c in categories.items()
        ),
        "drills": "\n".join(f"- {d['id']}: {d['category']}: {d['name']}" for d in drills),
        "category_names": ", ".join(categories),
        "drill_names": ", ".join(f"{d['id']} ({d['category']})" for d in drills),
    }
    user = Template(prompts["user"]).substitute(fields)
    images = [analysis.images[p] for p in stroke.PHASES]
    allowed = None
    if prompts.get("constrain_to_evidence"):
        allowed = evidence_categories(criteria, analysis.measurements)
    schema = feedback_schema(stroke, allowed)
    result = vlm.generate_json(prompts["system"], user, images, schema, **generation)
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
