"""B0, the rule baseline of the Final Project: the tossing arm drops early when the tossing wrist
is already at shoulder height or below too long before contact.

    uv run python -m strokes.serve.baseline

Reads data/processed/serve_dataset.npz (python -m strokes.serve.dataset). The only parameter,
the time threshold, is the one with the best macro-F1 on the train split; it is then scored on
the test split. The trained models (M1, M2) must beat it on the same split and metrics.
"""

import numpy as np

from eval.run_eval import macro_prf
from strokes.serve import dataset

POSITIVE, NEGATIVE = "toss_arm_drops_early", "no_clear_problem"


def drop_time(x: np.ndarray) -> float:
    """Seconds before contact the tossing wrist was last above its shoulder, in one window of
    the dataset. The racket hand is the higher wrist at contact; image y grows downward."""
    before = dataset.BEFORE
    left, right = (dataset.JOINTS.index(f"{side}_wrist") for side in ("left", "right"))
    side = "right" if x[before, left, 1] < x[before, right, 1] else "left"
    wrist, shoulder = (dataset.JOINTS.index(f"{side}_{joint}") for joint in ("wrist", "shoulder"))
    above = np.flatnonzero(x[: before + 1, wrist, 1] < x[: before + 1, shoulder, 1])
    return (before - above[-1] if len(above) else before + 1) / dataset.FPS


def auroc(positive: np.ndarray, score: np.ndarray) -> float:
    """Chance that a positive scores above a negative, ties counted half."""
    pos, neg = score[positive][:, None], score[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def predict(score: np.ndarray, threshold: float) -> list[str]:
    return np.where(score >= threshold, POSITIVE, NEGATIVE).tolist()


def main() -> None:
    data = np.load(dataset.OUT)
    labeled = data["valid"] == 1
    score = np.array([drop_time(x) for x in data["x"]])
    truth = data["category"]
    train = labeled & (data["split"] == "train")
    threshold = max(np.unique(score[train]), key=lambda t: macro_prf(
        truth[train].tolist(), predict(score[train], t))[2])  # fmt: skip
    print(f"B0: tossing arm drops early when it is down {threshold:.2f} s or more before contact")
    for split in ("train", "test"):
        chosen = labeled & (data["split"] == split)
        f1 = macro_prf(truth[chosen].tolist(), predict(score[chosen], threshold))[2]
        positive = truth[chosen] == POSITIVE
        print(f"  {split}: {chosen.sum()} videos ({positive.sum()} drop early), macro-F1 {f1:.3f}, "
              f"AUROC {auroc(positive, score[chosen]):.3f}")  # fmt: skip
    chosen = labeled & (data["split"] == "test")
    wrong = [f"{v} ({s:.2f} s, {c})" for v, s, c, p in zip(data["video"][chosen], score[chosen],
             truth[chosen], predict(score[chosen], threshold)) if c != p]  # fmt: skip
    print("  test errors:", ", ".join(wrong) or "none")


if __name__ == "__main__":
    main()
