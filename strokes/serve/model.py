"""A trained serve model at work: whether the tossing arm drops early in one video, how far the
seeds agree, and which arm joints and moments the answer rests on (occlusion).

    uv run python -m strokes.serve.model [--name m1c]

Used by the app (app/service.py). Loads experiments/serve/<name>.pt (python -m
strokes.serve.train) and the threshold eval/value.py chose for it on validation scores. The CLI
writes the occlusion map of the written report (section 4.7) over the test videos to
experiments/serve/<name>-occlusion.csv/.png.

Occlusion replaces one arm joint during one stretch of BIN frames by the same joint of a clean
serve, the mean window of the train serves labeled no_clear_problem (a baseline replacement, as in
SHAP), and measures how much the probability of "drops early" falls: how much that part of the
motion, against a clean serve, pushes the answer. Hiding the points instead, as MediaPipe does when
it loses one (mask 0), mostly measures the mask channel: an unseen point raises the probability.
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from strokes.serve import baseline, dataset, train

BIN = 8  # frames of one occluded stretch: 0.27 s at 30 fps
STARTS = np.arange(0, dataset.BEFORE + 1 + dataset.AFTER, BIN)  # 8 stretches of the 64 frames
STANCE = 16  # first frames of the window (0.53 s): the player stands, before the toss
# Shoulder width (shoulder_width) under which the app asks for a new recording. Our 44 videos: 0.17
# to 0.36 in 5 the team could not judge, 0.47 or more in every judgeable one; 79 % of the THETIS
# serves, filmed from in front of the player, are under it.
MIN_SHOULDERS = 0.4
# The joints of train.ARMS after train.arms: the tossing arm is the left one
PARTS = [
    f"{arm} {joint}" for arm in ("tossing", "racket") for joint in ("shoulder", "elbow", "wrist")
]
HEIGHT_PARTS = [p for p in PARTS if "shoulder" not in p]  # train.heights: above each shoulder


def load(name: str) -> dict:
    """The models of every seed, their clean reference serve, and the threshold eval/value.py chose
    and the test rates at it (experiments/serve/value.csv)."""
    saved = torch.load(baseline.RESULTS / f"{name}.pt")
    inputs = saved["weights"][0]["0.weight"].shape[1]  # of the first convolution
    heights = saved.get("heights", False)
    models = []
    for weights in saved["weights"]:
        model = train.network(inputs)
        model.load_state_dict(weights)
        models.append(model.eval())
    rows = csv.DictReader((baseline.RESULTS / "value.csv").open())
    row = next(r for r in rows if r["model"] == name)
    parts = HEIGHT_PARTS if heights else PARTS if saved["arms"] else None  # of occlusion maps
    return {"name": name, "arms": saved["arms"], "heights": heights, "parts": parts,
            "models": models, "reference":
            saved["reference"].numpy(), **{k: float(row[k]) for k in ("threshold", "test_tpr",
            "test_fpr")}}  # fmt: skip


def inputs(model: dict, x: np.ndarray, mask: np.ndarray):
    if model["heights"]:
        return train.heights(x, mask)
    return train.arms(x, mask) if model["arms"] else (x, mask)


def probabilities(model: dict, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(seeds, windows) probability that the tossing arm drops early, on model inputs."""
    return np.array([train.probability(m, x, mask) for m in model["models"]])


def replace(x: np.ndarray, reference: np.ndarray, joint: int, start: int) -> np.ndarray:
    """A copy of (windows, frames, joints, 2) x where `joint` moves like the reference serve in
    frames start..start + BIN."""
    x = x.copy()
    x[:, start : start + BIN, joint] = reference[start : start + BIN, joint]
    return x


def occlusion(model: dict, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(windows, joints, stretches): mean fall over seeds of the probability when one joint moves
    like the clean reference during one stretch, on model inputs."""
    base = probabilities(model, x, mask).mean(0)
    fall = np.zeros((len(x), x.shape[2], len(STARTS)))
    for j in range(x.shape[2]):
        for k, start in enumerate(STARTS):
            hidden = replace(x, model["reference"], j, start)
            fall[:, j, k] = base - probabilities(model, hidden, mask).mean(0)
    return fall


def seconds(start: int) -> float:
    """Time of a frame of the window, in seconds from contact."""
    return (start - dataset.BEFORE) / dataset.FPS


def shoulder_width(x: np.ndarray) -> np.ndarray:
    """Median distance across the image between the shoulders in the first STANCE frames of each
    window, in torso lengths: about 0.6 with the camera to the side looking at the chest, as the
    app asks; about 0.2 with the camera in front of or behind the player (THETIS)."""
    left, right = dataset.SHOULDERS
    return np.median(np.abs(x[..., :STANCE, left, 0] - x[..., :STANCE, right, 0]), -1)


def predict(model: dict, keypoints: np.ndarray, fps: float) -> dict:
    """The answer for one video's landmarks (core.pose.extract). Raises ValueError when the
    video shows no toss."""
    frame = dataset.contact(keypoints, fps)
    x, mask = dataset.window(keypoints, fps, frame)
    x, mask = x[None], mask[None]
    side = baseline.tossing_side(x[0])
    mx, mm = inputs(model, x, mask)
    seeds = probabilities(model, mx, mm)[:, 0]
    fall = occlusion(model, mx, mm)[0] if model["parts"] else None
    return {
        "probability": float(seeds.mean()),
        "seeds": seeds.round(4).tolist(),
        "threshold": model["threshold"],
        "drops_early": bool(seeds.mean() >= model["threshold"]),
        "tossing_side": side,
        "drop_seconds": baseline.drop_time(x[0]),  # B0's measure, shown as evidence
        "seen": float(train.arms(x, mask)[1].mean()),  # share of arm points MediaPipe saw
        "shoulders": float(shoulder_width(x[0])),
        "contact": frame,
        "occlusion": None if fall is None else fall.round(4).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="m1c")
    args = parser.parse_args()
    model = load(args.name)
    if not model["parts"]:
        raise SystemExit("occlusion maps are for arm models (M1b, M1c, M1e, M2)")
    data = np.load(dataset.OUT)
    test = (data["valid"] == 1) & (data["split"] == "test")
    positive = data["category"][test] == baseline.POSITIVE
    fall = occlusion(model, *inputs(model, data["x"][test], data["mask"][test]))
    rows = [{"model": args.name, "class": c, "part": part, "start_s": round(seconds(s), 3),
             "fall": round(float(fall[chosen, j, k].mean()), 5)}
            for c, chosen in ((baseline.POSITIVE, positive), (baseline.NEGATIVE, ~positive))
            for j, part in enumerate(model["parts"]) for k, s in enumerate(STARTS)]  # fmt: skip
    baseline.save(f"{args.name}-occlusion", rows)
    _, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    groups = (("drops early", positive), ("no problem", ~positive))
    limit = max(np.abs(fall[chosen].mean(0)).max() for _, chosen in groups)
    for ax, (title, chosen) in zip(axes, groups):
        image = ax.imshow(fall[chosen].mean(0), cmap="RdBu_r", vmin=-limit, vmax=limit,
                          aspect="auto")  # fmt: skip
        ax.set_xticks(range(len(STARTS)), [f"{seconds(s):+.1f}" for s in STARTS])
        ax.set(
            title=f"{title} ({chosen.sum()} test videos)", xlabel="stretch start (s from contact)"
        )
    axes[0].set_yticks(range(len(model["parts"])), model["parts"])
    plt.colorbar(image, ax=axes, label="fall in P(drops early)")
    plt.suptitle(f"{args.name.upper()}: what the answer rests on (occlusion)", y=1.04)
    plt.savefig(baseline.RESULTS / f"{args.name}-occlusion.png", dpi=150, bbox_inches="tight")
    top = np.unravel_index(np.abs(fall[positive].mean(0)).argmax(), fall.shape[1:])
    print(f"{args.name.upper()}: on 'drops early' test videos the answer rests most on the "
          f"{model['parts'][top[0]]} from {seconds(STARTS[top[1]]):+.2f} s")  # fmt: skip


if __name__ == "__main__":
    main()
