"""M2's pretraining: the encoder of M1b learns how arms move in a serve without labels, by filling
in stretches of the arms that are hidden from it.

    uv run python -m strokes.serve.pretrain [--seeds 5] [--epochs 300]

Data: THETIS (github.com/THETIS-dataset, Gourgari et al. 2013), 489 flat, kick and slice serves
of 55 players filmed indoors from the front, and the windows of the train split of
serve_dataset.npz with their labels unused; the test split is never seen. THETIS is prepared once,
in ignored folders:

    uv run python -m core.pose data/external/thetis/*.avi --out data/processed/thetis-pose
    uv run python -m strokes.serve.dataset --pose data/processed/thetis-pose \\
        --out data/processed/thetis_dataset.npz

Each epoch hides about HIDDEN of the frames, in stretches of SPAN, from the input (position and
seen set to 0) and the encoder plus a 1 × 1 convolution must give back the arm positions there;
the loss is the squared error on the hidden points MediaPipe saw. Every fifth THETIS player is held
out to check that the reconstruction generalizes. Writes experiments/serve/pretrain.pt (the
encoder of each seed, for `train.py --init pretrain`) and pretrain-curves.csv/.png.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from core import ROOT
from strokes.serve import baseline, dataset, train

THETIS = ROOT / "data" / "processed" / "thetis_dataset.npz"
SPAN, HIDDEN = 8, 0.3  # stretches of 8 frames, about 30 % of the frames hidden


def hide(x: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Copies of x and mask with random stretches set to 0, and the (windows, frames) hidden."""
    hidden = np.zeros(x.shape[:2], bool)
    for row in hidden:
        for start in rng.choice(x.shape[1] - SPAN, round(HIDDEN * x.shape[1] / SPAN)):
            row[start : start + SPAN] = True
    x, mask = x.copy(), mask.copy()
    x[hidden], mask[hidden] = 0, False
    return x, mask, hidden


def reconstruction(model: nn.Module, x, mask, rng) -> torch.Tensor:
    """Mean squared error on the hidden points that were seen."""
    shown, shown_mask, hidden = hide(x, mask, rng)
    out = model(train.features(shown, shown_mask))  # (windows, 2 × joints, frames)
    target = train.features(x, mask)[:, : out.shape[1]]
    weight = torch.from_numpy(np.repeat(mask, 2, axis=2) & hidden[:, :, None]).transpose(1, 2)
    return ((out - target) ** 2 * weight).sum() / weight.sum().clamp(min=1)


def interpolation(x: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> float:
    """The same error for a straight line across each hidden stretch, with the stretches hide()
    draws from rng: the encoder must beat it to have learned more than smooth motion."""
    _, _, hidden = hide(x, mask, rng)
    frames, error, count = np.arange(x.shape[1]), 0.0, 0
    for w, row in enumerate(hidden):
        for j in range(x.shape[2]):
            chosen = row & mask[w, :, j]
            for c in range(2):
                guess = np.interp(frames[chosen], frames[~row], x[w, ~row, j, c])
                error += ((guess - x[w, chosen, j, c]) ** 2).sum()
            count += 2 * chosen.sum()
    return error / count


def pretrain(x, mask, held_x, held_mask, seed: int, epochs: int):
    """Full-batch AdamW on the reconstruction loss. Returns the encoder's weights and, per epoch,
    the loss on the windows used and on the held-out players (same hidden stretches each epoch)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder = train.network(3 * x.shape[2])[: train.ENCODER]
    model = nn.Sequential(encoder, nn.Conv1d(encoder[3].out_channels, 2 * x.shape[2], 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    curve = []
    for _ in range(epochs):
        model.train()
        error = reconstruction(model, *train.augment(x, mask, rng, flip=False), rng)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            held = reconstruction(model, held_x, held_mask, np.random.default_rng(0)).item()
        curve.append([error.item(), held])
    return encoder.state_dict(), curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    thetis, ours = np.load(THETIS), np.load(dataset.OUT)
    players = np.unique(thetis["player"])
    held = np.isin(thetis["player"], players[::5])
    mine = ours["split"] == "train"
    x, mask = train.arms(np.concatenate([thetis["x"][~held], ours["x"][mine]]),
                         np.concatenate([thetis["mask"][~held], ours["mask"][mine]]))  # fmt: skip
    held_x, held_mask = train.arms(thetis["x"][held], thetis["mask"][held])
    line = interpolation(held_x, held_mask, np.random.default_rng(0))
    print(f"Pretraining on {len(x)} windows ({(~held).sum()} THETIS, {mine.sum()} of the train "
          f"split); {held.sum()} windows of {len(players[::5])} THETIS players held out, where a "
          f"straight line across each hidden stretch errs {line:.3f}")  # fmt: skip
    encoders, curves = [], []
    for seed in range(args.seeds):
        weights, curve = pretrain(x, mask, held_x, held_mask, seed, args.epochs)
        encoders.append(weights)
        curves += [{"seed": seed, "epoch": e, "train": t, "held_out": h}
                   for e, (t, h) in enumerate(curve, 1)]  # fmt: skip
        print(f"  seed {seed}: reconstruction {curve[0][1]:.3f} at epoch 1, {curve[-1][1]:.3f} at "
              f"epoch {args.epochs} on held-out players")  # fmt: skip
    torch.save({"encoders": encoders, "epochs": args.epochs}, baseline.RESULTS / "pretrain.pt")
    baseline.save("pretrain-curves", curves)
    plot(curves, line)


def plot(curves: list[dict], line: float) -> None:
    """Mean reconstruction error per epoch over seeds, against the straight line."""
    _, ax = plt.subplots(figsize=(6, 3.5))
    for split in ("train", "held_out"):
        loss = np.array([[c[split] for c in curves if c["seed"] == s]
                         for s in sorted({c["seed"] for c in curves})])  # fmt: skip
        ax.plot(np.arange(1, loss.shape[1] + 1), loss.mean(0), label=split.replace("_", "-"))
    ax.axhline(line, color="grey", ls="--", label="straight line, held-out")
    ax.set(xlabel="epoch", ylabel="squared error, torso lengths²", yscale="log",
           title="M2 pretraining: filling in hidden stretches of the arms")  # fmt: skip
    ax.legend()
    plt.tight_layout()
    plt.savefig(baseline.RESULTS / "pretrain-curves.png", dpi=150)


if __name__ == "__main__":
    main()
