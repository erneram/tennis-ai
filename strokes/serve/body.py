"""The serve beyond the tossing arm: the team's answers of the "Cuerpo" tab (eval/label.py) as
targets, and one pose measure per answer that needs no training.

Every window is first turned so the tossing arm is the left one and x grows toward the court
(players land into it), so a measure reads the same for every player and camera side. Each
measure is signed so that a higher value means more to improve; the sign is fixed here from what
the question asks, not fitted to the answers (eval/body.py scores how well each one follows them).
"""

import csv
from pathlib import Path

import numpy as np

from strokes.serve import baseline, dataset, train

J = dataset.JOINTS.index
LOAD = 30  # frames before contact the legs bend in: 1 s
NEAR = 3  # frames on each side of contact for the reach: 0.1 s, the contact frame is found early
# Answers that mean "needs work" (1) and "as it should be" (0); the rest ("unseen", "too_much")
# are left out. Where the ball and contact point are cannot be read from the pose, so
# contact_point has no target (its ideal, "ahead", is also not what the professionals got).
ANSWERS = {
    "legs": ({"none", "little"}, {"good"}),
    "shoulders": ({"none", "little"}, {"good"}),
    "racket_at_peak": ({"low", "rising", "dropped", "hitting"}, {"trophy"}),
    "racket_drop": ({"none", "little"}, {"good"}),
    "arm_at_contact": ({"bent", "forward"}, {"up"}),
    "landing": ({"in_place", "off_balance"}, {"forward"}),
}
MEASURES = {
    "legs": "hips drop in the second before contact, negated (torso lengths)",
    "shoulders": "tossing shoulder above the racket shoulder at the trophy, negated (torso)",
    "racket_at_peak": "seconds from the racket wrist lowest after the trophy to contact",
    "racket_drop": "racket wrist above its shoulder at its lowest after the trophy (torso)",
    "arm_at_contact": "highest racket wrist above its shoulder within 0.1 s of contact, negated",
    "landing": "hips forward in the 0.5 s after contact, negated (torso lengths)",
}


def oriented(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copies of the windows with the tossing arm on the left and x toward the court."""
    x, mask = x.copy(), mask.copy()
    train.mirror(x, mask, np.array([baseline.tossing_side(w) == "right" for w in x], bool))
    hips = x[:, :, dataset.HIPS, 0].mean(2)
    x[..., 0] *= np.where(hips[:, -1] >= hips[:, 0], 1, -1)[:, None, None]
    return x, mask


def measures(x: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """One value per window and aspect of MEASURES, higher meaning more to improve."""
    x, _ = oriented(x, mask)
    y = x[..., 1]  # image y grows downward
    before, rows = dataset.BEFORE, np.arange(len(x))
    hips = x[:, :, dataset.HIPS].mean(2)
    racket = y[:, :, J("right_shoulder")] - y[:, :, J("right_wrist")]  # above its shoulder
    trophy = (y[:, :before, J("left_shoulder")] - y[:, :before, J("left_wrist")]).argmax(1)
    # The racket drops behind the back between the trophy (tossing wrist highest) and contact
    after_trophy = np.arange(before)[None] >= trophy[:, None]
    low = np.where(after_trophy, racket[:, :before], np.inf).argmin(1)
    tilt = y[:, :, J("right_shoulder")] - y[:, :, J("left_shoulder")]
    return {
        "legs": np.median(hips[:, :10, 1], 1) - hips[:, before - LOAD : before, 1].max(1),
        "shoulders": -tilt[rows, trophy],
        "racket_at_peak": (before - low) / dataset.FPS,
        "racket_drop": racket[rows, low],
        "arm_at_contact": -racket[:, before - NEAR : before + NEAR + 1].max(1),
        "landing": hips[:, before, 0] - hips[:, -1, 0],
    }


def answers(path: Path) -> dict[str, dict]:
    """Rows of an aspects_<name>.csv with every question answered, by video name without suffix."""
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    return {Path(r["video"]).stem: r for r in rows if r.get("legs")}


def targets(answered: dict[str, dict], videos) -> np.ndarray:
    """(N, len(ANSWERS)): 1 needs work, 0 as it should be, NaN not answered or left out."""
    y = np.full((len(videos), len(ANSWERS)), np.nan, np.float32)
    for i, video_name in enumerate(videos):
        row = answered.get(str(video_name))
        for k, (aspect, (needs, fine)) in enumerate(ANSWERS.items()):
            if row and row[aspect] in needs | fine:
                y[i, k] = row[aspect] in needs
    return y
