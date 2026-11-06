"""M1, the first trained model of the Final Project: a small temporal convolutional network,
trained from scratch, that says whether the tossing arm drops early.

    uv run python -m strokes.serve.train [--seeds 5] [--epochs 300] [--arms]

Reads data/processed/serve_dataset.npz (python -m strokes.serve.dataset). Input: the 64 frames
of a window, each with x, y and seen/not seen of the 17 joints (51 channels). Trained on the
labeled train split with random mirroring (left and right joints swapped), scaling, noise and a
shift of a few frames, then scored on the test split with the metrics of B0
(strokes/serve/baseline.py): macro-F1 at probability 0.5 and AUROC, mean and SD over seeds.
Hyperparameters are fixed in advance: the train split is too small to hold out a validation set.
--arms trains M1b: only the shoulders, elbows and wrists, mirrored so the tossing arm is always
the left one (and never mirrored at random), so 30 videos fit fewer weights.
"""

import argparse

import numpy as np
import torch
from torch import nn

from eval.run_eval import macro_prf
from strokes.serve import baseline, dataset

# Index of the same joint on the other side of the body, to mirror a window
MIRROR = [
    dataset.JOINTS.index(j.replace("left", "@").replace("right", "left").replace("@", "right"))
    for j in dataset.JOINTS
]
ARMS = [dataset.JOINTS.index(f"{side}_{joint}")
        for side in ("left", "right") for joint in ("shoulder", "elbow", "wrist")]  # fmt: skip
SHIFT, SCALE, NOISE = 3, 0.1, 0.02  # frames, relative size, torso lengths


def network(inputs: int = 3 * len(dataset.JOINTS), channels: int = 32, dropout: float = 0.3):
    """Two 1-D convolutions over time (the second dilated), max over time, one output logit."""
    return nn.Sequential(
        nn.Conv1d(inputs, channels, 5, padding=2), nn.ReLU(), nn.Dropout(dropout),
        nn.Conv1d(channels, channels, 5, padding=4, dilation=2), nn.ReLU(),
        nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, 1),
    )  # fmt: skip


def features(x: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """(N, T, J, 2) points and (N, T, J) mask as (N, 3J, T) channels over time."""
    n, t = x.shape[:2]
    both = np.concatenate([x.reshape(n, t, -1), mask.reshape(n, t, -1)], axis=2)
    return torch.from_numpy(both.astype(np.float32)).transpose(1, 2)


def mirror(x: np.ndarray, mask: np.ndarray, chosen: np.ndarray) -> None:
    """Swap the left and right joints of the chosen windows and negate their x, in place."""
    x[chosen] = x[chosen][:, :, MIRROR] * [-1, 1]
    mask[chosen] = mask[chosen][:, :, MIRROR]


def arms(x: np.ndarray, mask: np.ndarray):
    """The arm joints only, with the tossing arm always on the left (input of M1b)."""
    x, mask = x.copy(), mask.copy()
    mirror(x, mask, np.array([baseline.tossing_side(w) == "right" for w in x], bool))
    return x[:, :, ARMS], mask[:, :, ARMS]


def augment(x: np.ndarray, mask: np.ndarray, rng: np.random.Generator, flip: bool = True):
    x, mask = x.copy(), mask.copy()
    chosen = rng.random(len(x)) < 0.5
    if flip:
        mirror(x, mask, chosen)
    x *= rng.uniform(1 - SCALE, 1 + SCALE, (len(x), 1, 1, 1))
    x += rng.normal(0, NOISE, x.shape)
    for i, shift in enumerate(rng.integers(-SHIFT, SHIFT + 1, len(x))):
        x[i], mask[i] = np.roll(x[i], shift, 0), np.roll(mask[i], shift, 0)
    return x, mask


def train(x, mask, y, seed: int, epochs: int, flip: bool = True) -> nn.Module:
    """Full-batch AdamW on the class-balanced binary cross-entropy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = network(3 * x.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    target = torch.tensor(y, dtype=torch.float32)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - y).sum() / y.sum()))
    for _ in range(epochs):
        model.train()
        error = loss(model(features(*augment(x, mask, rng, flip))).squeeze(1), target)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def probability(model: nn.Module, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return torch.sigmoid(model(features(x, mask)).squeeze(1)).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--arms", action="store_true", help="train M1b on the arms only")
    args = parser.parse_args()
    data = np.load(dataset.OUT)
    x, mask = arms(data["x"], data["mask"]) if args.arms else (data["x"], data["mask"])
    labeled = data["valid"] == 1
    split = {s: labeled & (data["split"] == s) for s in ("train", "test")}
    truth = data["category"]
    y = (truth[split["train"]] == baseline.POSITIVE).astype(np.float32)
    weights = sum(p.numel() for p in network(3 * x.shape[2]).parameters())
    print(f"{'M1b' if args.arms else 'M1'}: {weights} weights, "
          f"{split['train'].sum()} train videos ({int(y.sum())} drop early)")  # fmt: skip
    results = {s: [] for s in split}
    for seed in range(args.seeds):
        chosen = split["train"]
        model = train(x[chosen], mask[chosen], y, seed, args.epochs, flip=not args.arms)
        line = []
        for name, chosen in split.items():
            score = probability(model, x[chosen], mask[chosen])
            f1 = macro_prf(truth[chosen].tolist(), baseline.predict(score, 0.5))[2]
            area = baseline.auroc(truth[chosen] == baseline.POSITIVE, score)
            results[name].append((f1, area))
            line.append(f"{name} macro-F1 {f1:.3f} AUROC {area:.3f}")
        print(f"  seed {seed}: " + ", ".join(line))
    for name, values in results.items():
        (f1, sd_f1), (area, sd_area) = zip(np.mean(values, 0), np.std(values, 0))
        print(f"  {name}: macro-F1 {f1:.3f} ± {sd_f1:.3f}, AUROC {area:.3f} ± {sd_area:.3f}")


if __name__ == "__main__":
    main()
