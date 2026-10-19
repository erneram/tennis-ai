"""Serve phase segmentation from a side view.

Events, in order (y grows downwards, so "highest" means smallest y):
- contact: the racket wrist reaches its highest point.
- trophy: the toss wrist reaches its highest point before contact.
- toss: the toss wrist is at its lowest point before trophy (the arm starts rising).
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
MIN_FOLLOW_SECONDS = 0.3


def racket_side(xy: np.ndarray) -> str:
    """'right' or 'left': the wrist that reaches the highest point holds the racket."""
    highest = {side: np.nanmin(point(xy, f"{side}_wrist")[:, 1]) for side in ("right", "left")}
    return min(highest, key=highest.get)


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
    if contact < 2:
        raise InsufficientEvidence(
            "El video empieza en el impacto; graba desde la posición inicial."
        )
    if (len(xy) - 1 - contact) / fps < MIN_FOLLOW_SECONDS:
        raise InsufficientEvidence("El video termina en el impacto; graba hasta la terminación.")

    trophy = int(np.nanargmin(toss_y[:contact]))
    toss_start = int(np.nanargmax(toss_y[: trophy + 1]))
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
