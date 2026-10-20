"""Serve measurements, computed from the cleaned pose at the key events.

Only measurements that hold up from the recording angle (camera behind the player) are kept:
vertical distances, arms pointing up, and timing. Angles that bend toward the net (knees,
racket elbow, arm at contact) look shorter than they are from behind, so they were removed
after testing on the first recordings; those categories are judged from the frames instead.

Distances are divided by the torso length, so they do not depend on camera distance.
"""

import numpy as np

from core.metrics import angle, point, torso_length
from strokes.serve.phases import racket_side

MEASUREMENT_LABELS = {
    "toss_arm_extension_deg": "Extensión del brazo de lanzamiento (°)",
    "toss_arm_elevation_deg": "Elevación del brazo de lanzamiento (°)",
    "racket_elbow_height": "Altura del codo de la raqueta sobre el hombro (torsos)",
    "toss_to_contact_s": "Tiempo del lanzamiento al impacto (s)",
}


def measure(xy: np.ndarray, fps: float, phases: dict[str, int]) -> dict[str, float]:
    racket = racket_side(xy)
    toss = "left" if racket == "right" else "right"
    trophy = phases["trophy"]

    def joint(name: str) -> np.ndarray:
        return point(xy, name)[trophy]

    values = {
        # Toss arm at trophy: straight (180) and pointing up (180 between torso and arm).
        "toss_arm_extension_deg": angle(
            joint(f"{toss}_shoulder"), joint(f"{toss}_elbow"), joint(f"{toss}_wrist")
        ),
        "toss_arm_elevation_deg": angle(
            joint(f"{toss}_hip"), joint(f"{toss}_shoulder"), joint(f"{toss}_wrist")
        ),
        # Racket elbow at trophy: height above the shoulder, in torso lengths.
        "racket_elbow_height": (joint(f"{racket}_shoulder")[1] - joint(f"{racket}_elbow")[1])
        / torso_length(xy),
        "toss_to_contact_s": (phases["contact"] - phases["toss"]) / fps,
    }
    return {name: round(float(value), 2) for name, value in values.items()}
