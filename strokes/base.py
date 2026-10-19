"""Interface every stroke implements, plus helpers shared by all strokes.

A stroke is a package strokes/<name>/ whose __init__.py exposes the Stroke attributes below,
with criteria.yaml, drills.yaml and prompts/<version>.yaml next to it. Adding a stroke means
adding a folder with the same structure; nothing in core/ changes.
"""

from importlib import import_module
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

STROKES_DIR = Path(__file__).resolve().parent


class Stroke(Protocol):
    NAME: str
    PHASES: tuple[str, ...]  # in chronological order
    LABELS: dict[str, str]  # phase -> name shown to the player

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
