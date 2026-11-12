"""How far JC's serve moved toward the professionals' after the feedback (written report, results).

    uv run python -m eval.improvement

JC's serves before the feedback (train and test splits) against his serves of Nov 7, recorded after
he was told what to change (split new), with EA and the professionals (pro01 to pro05) as
reference, by three measures:

1. Toss height: the labeling question as a number, the height of the tossing wrist above its
   shoulder (torso lengths) at the frame the racket wrist is lowest in the 0.6 s before contact.
   No training and no labels. "In the professionals' range" is the share of serves at or above
   the professionals' 10th percentile of it.
2. The app's answer: the share of serves the app's model sends to the drill, from its validation or
   test scores for the serves it was trained or chosen on and from the frozen model for the rest.
3. The team's labels: the share with the fault. The labels after the feedback are not blind.

Gap closed = (after - before) / (professionals - before): 0 % no change, 100 % like the
professionals. The 95 % interval resamples serves; all serves after the feedback come from one
session, so it is too narrow. Writes experiments/serve/improvement.csv.
"""

import numpy as np

from eval import bias
from strokes.serve import baseline, dataset, model, train

MODEL = "m1e"  # the app's model (configs/default.yaml)
LOOK = train.LOOK  # frames before contact the racket wrist is searched at its lowest
ROUNDS = 2000
GROUPS = ["JC before", "JC after", "EA", "professionals"]


def toss_height(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Height of the tossing wrist above its shoulder when the racket wrist is lowest, per window."""
    up = train.heights(x, mask)[0][..., 0]  # tossing elbow, wrist, racket elbow, wrist
    low = dataset.BEFORE - LOOK + up[:, dataset.BEFORE - LOOK : dataset.BEFORE, 3].argmin(1)
    return up[np.arange(len(up)), low, 1]


def toss_probability(data) -> np.ndarray:
    """The model's mean probability over its seeds per serve: its validation or test score where
    it has one, else the frozen model's (on the serves it was never trained on)."""
    loaded = model.load(MODEL)
    frozen = model.probabilities(loaded, *model.inputs(loaded, data["x"], data["mask"])).T
    scored = {video: score for (_, video), score in bias.scores(MODEL).items()}
    return np.array([scored.get(v, f).mean() for v, f in zip(data["video"], frozen)])


def drills(data) -> np.ndarray:
    """Whether the model sends each serve to the drill, as the app does: the mean probability over
    its seeds at or above its threshold."""
    return toss_probability(data) >= bias.thresholds()[MODEL]


def gap_closed(before: np.ndarray, after: np.ndarray, pros: np.ndarray, stat) -> float:
    return float((stat(after) - stat(before)) / (stat(pros) - stat(before)))


def group(video: str, player: str) -> str:
    if video.startswith("jc-post"):
        return "JC after"
    if player == "jc":
        return "JC before"
    return "professionals" if player.startswith("pro") else player.upper()


def main() -> None:
    data = np.load(dataset.OUT)
    judged = data["valid"] == 1
    groups = np.array([group(v, p) for v, p in zip(data["video"], data["player"])])[judged]
    height = toss_height(data["x"], data["mask"])[judged]
    floor = np.percentile(height[groups == "professionals"], 10)
    measures = {
        "toss height (median, torso lengths)": (height, np.median),
        "in the professionals' range": (height >= floor, np.mean),
        f"sent to the drill by {MODEL.upper()}": (drills(data)[judged], np.mean),
        "fault in the team's labels": (data["category"][judged] == baseline.POSITIVE, np.mean),
    }
    rng = np.random.default_rng(0)
    rows = []
    for name, (values, stat) in measures.items():
        mine = {g: values[groups == g] for g in GROUPS}
        for g in GROUPS:
            rows.append({"measure": name, "group": g, "serves": len(mine[g]),
                         "value": round(float(stat(mine[g])), 3), "low": "", "high": ""})  # fmt: skip
        ends = mine["JC before"], mine["JC after"], mine["professionals"]
        boot = [gap_closed(*(rng.choice(v, len(v)) for v in ends), stat) for _ in range(ROUNDS)]
        low, high = np.percentile(boot, [2.5, 97.5])
        rows.append({"measure": name, "group": "gap closed", "serves": "",
                     "value": round(gap_closed(*ends, stat), 3),
                     "low": round(low, 3), "high": round(high, 3)})  # fmt: skip
        print(f"{name}: " + ", ".join(f"{r['group']} {r['value']}" for r in rows[-5:-1])
              + f"; gap closed {rows[-1]['value']:.0%} ({low:.0%} to {high:.0%})")  # fmt: skip
    print(f"  (professionals' 10th percentile of the toss height: {floor:.2f})")
    baseline.save("improvement", rows)


if __name__ == "__main__":
    main()
