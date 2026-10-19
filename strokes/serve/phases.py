"""Serve phase segmentation from pose (y grows downwards, so "highest" means smallest y).

Each wrist peaks once in a serve: the toss arm first (trophy), the racket arm about one second
later (contact). The order of the two peaks tells which arm holds the racket, which also works
when the pose model mirrors left and right for a player seen from behind.

Events:
- contact: the racket wrist reaches its highest point.
- trophy: the toss wrist reaches its highest point in the 2 s before contact.
- toss: the toss wrist is at its lowest point in the 1.2 s before trophy (the arm starts rising).
- leg_drive: deepest knee bend between trophy and contact (the push-off starts).
- stance: from the start of the clip; follow_through: right after contact.
"""

import numpy as np

from core.analysis import InsufficientEvidence
from core.metrics import angle, point

PHASES = ("stance", "toss", "trophy", "leg_drive", "contact", "follow_through")
LABELS = {
    "stance": "1. Posición inicial",
    "toss": "2. Lanzamiento",
    "trophy": "3. Posición de trofeo",
    "leg_drive": "4. Impulso de piernas",
    "contact": "5. Impacto",
    "follow_through": "6. Terminación",
}
MAX_TROPHY_TO_CONTACT_S = 2.0
TOSS_WINDOW_S = 1.2
MIN_FOLLOW_S = 0.3


def racket_side(xy: np.ndarray) -> str:
    """'left' or 'right' as labeled by the pose model: the wrist whose highest point comes later."""
    peak = {side: int(np.nanargmin(point(xy, f"{side}_wrist")[:, 1])) for side in ("left", "right")}
    return max(peak, key=peak.get)


def knee_angle(xy: np.ndarray) -> np.ndarray:
    """Mean of both knee angles per frame (180 = straight legs)."""
    knees = [
        angle(point(xy, f"{s}_hip"), point(xy, f"{s}_knee"), point(xy, f"{s}_ankle"))
        for s in ("left", "right")
    ]
    return np.nanmean(knees, axis=0)


def segment(xy: np.ndarray, fps: float) -> dict[str, int]:
    for name in ("left_wrist", "right_wrist", "nose"):
        if np.isnan(point(xy, name)).all():
            raise InsufficientEvidence("No se ven los brazos o la cabeza del jugador.")

    racket = racket_side(xy)
    toss = "left" if racket == "right" else "right"
    racket_y = point(xy, f"{racket}_wrist")[:, 1]
    toss_y = point(xy, f"{toss}_wrist")[:, 1]

    contact = int(np.nanargmin(racket_y))
    if not racket_y[contact] < point(xy, "nose")[contact, 1]:
        raise InsufficientEvidence("No se detecta el impacto por encima de la cabeza.")
    if (len(xy) - 1 - contact) / fps < MIN_FOLLOW_S:
        raise InsufficientEvidence("El video termina en el impacto; graba hasta la terminación.")

    first = max(0, contact - round(MAX_TROPHY_TO_CONTACT_S * fps))
    if contact - first < 2:
        raise InsufficientEvidence(
            "El video empieza en el impacto; graba desde la posición inicial."
        )
    trophy = first + int(np.nanargmin(toss_y[first:contact]))
    first = max(0, trophy - round(TOSS_WINDOW_S * fps))
    toss_start = first + int(np.nanargmax(toss_y[first : trophy + 1]))
    knees = knee_angle(xy)[trophy : contact + 1]
    leg_drive = trophy + (int(np.nanargmin(knees)) if not np.isnan(knees).all() else 0)

    phases = dict(zip(PHASES, (0, toss_start, trophy, leg_drive, contact, contact + 1)))
    if not toss_start < trophy <= leg_drive <= contact:
        raise InsufficientEvidence(
            "Las fases del saque no aparecen en el orden esperado; graba de nuevo el saque completo."
        )
    return phases


def key_frames(phases: dict[str, int], n_frames: int, fps: float) -> dict[str, int]:
    frames = dict(phases)
    frames["stance"] = max(0, phases["toss"] - round(0.2 * fps))
    frames["follow_through"] = min(n_frames - 1, phases["contact"] + round(0.4 * fps))
    return frames
