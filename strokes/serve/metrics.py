"""Serve measurements, computed from the cleaned pose at the key events.

Only measurements that hold up from one camera and have a clear meaning are kept. Angles that
bend toward the net (knees, arm at contact) look shorter than they are from behind, and the
racket elbow height had no published limit, so those categories are judged from the frames
(see criteria.yaml).
"""

import numpy as np

from core.metrics import angle, point
from strokes.serve.phases import racket_side

MEASUREMENT_LABELS = {
    "toss_arm_extension_deg": "Extensión del brazo de lanzamiento (°)",
    "toss_to_contact_s": "Tiempo del lanzamiento al impacto (s)",
}


def measure(xy: np.ndarray, fps: float, phases: dict[str, int]) -> dict[str, float]:
    toss = "left" if racket_side(xy) == "right" else "right"
    trophy = phases["trophy"]

    def joint(name: str) -> np.ndarray:
        return point(xy, name)[trophy]

    values = {
        # Toss arm at the top of the toss: straight is 180.
        "toss_arm_extension_deg": angle(
            joint(f"{toss}_shoulder"), joint(f"{toss}_elbow"), joint(f"{toss}_wrist")
        ),
        "toss_to_contact_s": (phases["contact"] - phases["toss"]) / fps,
    }
    return {name: round(float(value), 2) for name, value in values.items()}
