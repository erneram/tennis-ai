"""Serve measurements, computed from the cleaned pose at the key events.

Distances are divided by the torso length, so they do not depend on camera distance.
The wrist stands in for the racket: the racket itself is not tracked.
"""

import numpy as np

from core.metrics import angle, point, torso_length
from strokes.serve.phases import knee_angle, racket_side


def measure(xy: np.ndarray, fps: float, phases: dict[str, int]) -> dict[str, float]:
    racket = racket_side(xy)
    toss = "left" if racket == "right" else "right"
    trophy, contact = phases["trophy"], phases["contact"]
    torso = torso_length(xy)

    def joint(name: str, frame: int) -> np.ndarray:
        return point(xy, name)[frame]

    def joint_angle(a: str, b: str, c: str, frame: int) -> float:
        return angle(joint(a, frame), joint(b, frame), joint(c, frame))

    def height_above(lower: str, upper: str, frame: int) -> float:
        """How far `upper` is above `lower`, in torso lengths."""
        return (joint(lower, frame)[1] - joint(upper, frame)[1]) / torso

    values = {
        # Toss arm at trophy: straight (180) and pointing up (180 between torso and arm).
        "toss_arm_extension_deg": joint_angle(
            f"{toss}_shoulder", f"{toss}_elbow", f"{toss}_wrist", trophy
        ),
        "toss_arm_elevation_deg": joint_angle(
            f"{toss}_hip", f"{toss}_shoulder", f"{toss}_wrist", trophy
        ),
        # Racket elbow at trophy: height above the shoulder and bend.
        "racket_elbow_height": height_above(f"{racket}_shoulder", f"{racket}_elbow", trophy),
        "racket_elbow_angle_deg": joint_angle(
            f"{racket}_shoulder", f"{racket}_elbow", f"{racket}_wrist", trophy
        ),
        # Knee bend: degrees away from straight legs at the deepest point before contact.
        "knee_bend_deg": 180 - np.nanmin(knee_angle(xy)[trophy : contact + 1]),
        # Contact: arm extension and wrist height above the head.
        "contact_arm_extension_deg": joint_angle(
            f"{racket}_shoulder", f"{racket}_elbow", f"{racket}_wrist", contact
        ),
        "contact_height": height_above("nose", f"{racket}_wrist", contact),
        "toss_to_contact_s": (contact - phases["toss"]) / fps,
    }
    return {name: round(float(value), 2) for name, value in values.items()}
