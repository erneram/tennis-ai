"""M1, the first trained model of the Final Project: a small temporal convolutional network,
trained from scratch, that says whether the tossing arm drops early.

    uv run python -m strokes.serve.train [--seeds 5] [--epochs 300] [--arms | --heights [--synthetic]] [--init pretrain] [--more jc-post] [--name m1]

Reads data/processed/serve_dataset.npz (python -m strokes.serve.dataset). Input: the 64 frames
of a window, each with x, y and seen/not seen of the 17 joints (51 channels). Trained on the
labeled train split with random mirroring (left and right joints swapped), scaling, noise and a
shift of a few frames, then scored with the metrics of B0 (strokes/serve/baseline.py): macro-F1
at probability 0.5 and AUROC, mean and SD over seeds. The train split is too small to hold out a
validation set, so the validation scores come from 5-fold cross-validation on it (each video
scored by the model that did not see it); hyperparameters are fixed in advance and no choice looks
at the test split. Writes the scores and the loss per epoch (train, validation, test) to
experiments/serve/<model>.csv and <model>-curves.csv/.png for eval/value.py and the report, and
the weights of the model of each seed trained on the whole train split, with the mean window of the
train serves without the problem, to <model>.pt for the app (strokes/serve/model.py).
--arms trains M1b: only the shoulders, elbows and wrists, mirrored so the tossing arm is always
the left one (and never mirrored at random), so 30 videos fit fewer weights. M1c is M1b stopped
at 85 epochs, where M1b's mean validation loss is lowest (experiments/serve/m1b-curves.png):
`--arms --epochs 85 --name m1c`. --init pretrain trains M2: M1b whose convolutions start from the
encoder strokes/serve/pretrain.py learned without labels (the same seed's), so the change against
M1b, trained from scratch the same way, is the effect of pretraining. --more adds the judged clips of
dataset.NEW whose names start with it to the train split (and to its cross-validation): M1d is M1c
with JC's serves after the feedback, all without the fault, so the change against M1c is the
effect of those data; EA stays unseen (`--arms --epochs 85 --more jc-post --name m1d`).
--heights trains M1e: M1c whose input is only the height of each elbow and wrist above its own
shoulder (`--heights --epochs 85 --name m1e`). Without the x of the points, which says where the
arms are across the image, and without the mask, the model sees what the labeling question asks
(is the tossing hand still up while the racket drops?) and less of what tells players and
recordings apart. M1f is M1e with JC's serves after the feedback, as M1d is M1c with them
(`--heights --epochs 85 --more jc-post --name m1f`).
--synthetic trains M1g: M1e with a copy of each serve without the fault where the tossing arm
drops early (synthetic), so serves with the fault are no longer only JC's: in each fold the copies
come from the fold's training serves only, and validation and test are scored on real serves
(`--heights --epochs 85 --synthetic --name m1g`).
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
FOLDS = 5
LOOK = 18  # frames before contact the racket wrist is searched at its lowest: 0.6 s
SOONER = 5  # the synthetic fault: the tossing wrist goes below its shoulder up to 5 frames earlier
ENCODER = 5  # network()[:ENCODER]: both convolutions, before pooling over time


def network(inputs: int = 3 * len(dataset.JOINTS), channels: int = 32, dropout: float = 0.3,
            outputs: int = 1):  # fmt: skip
    """Two 1-D convolutions over time (the second dilated), max over time, one logit per output."""
    return nn.Sequential(
        nn.Conv1d(inputs, channels, 5, padding=2), nn.ReLU(), nn.Dropout(dropout),
        nn.Conv1d(channels, channels, 5, padding=4, dilation=2), nn.ReLU(),
        nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, outputs),
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


def heights(x: np.ndarray, mask: np.ndarray):
    """(N, T, 4, 1) height above its own shoulder of the tossing elbow and wrist, then the racket
    elbow and wrist, in torso lengths, with every point counted as seen (input of M1e)."""
    y = arms(x, mask)[0][..., 1]  # image y grows downward
    up = y[:, :, [0, 0, 3, 3]] - y[:, :, [1, 2, 4, 5]]
    return up[..., None], np.ones(up.shape, bool)


def synthetic(up: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Copies of heights() windows of serves without the fault where the tossing arm drops early:
    its elbow and wrist are warped in time between the top of the toss and contact so that the
    wrist goes below its shoulder 0 to SOONER frames before the racket wrist is lowest, as the
    labeling question asks; the racket arm and everything after contact stay as they were. A
    serve where this cannot be done is skipped, so every copy has the fault."""
    before, frames, copies = dataset.BEFORE, np.arange(up.shape[1]), []
    for window in up:
        toss, racket = window[:, 1, 0], window[:, 3, 0]
        low = before - LOOK + racket[before - LOOK : before].argmin()
        top = toss[:before].argmax()
        under = np.flatnonzero(toss[top:before] < 0)
        early = low - rng.integers(0, SOONER + 1)
        if not len(under) or top < 1 or early <= top + 1 or top + under[0] <= early:
            continue  # no drop before contact, or the wrist already drops by then
        source = np.interp(frames, [0, top, early, before, frames[-1]],
                           [0, top, top + under[0], before, frames[-1]])  # fmt: skip
        warped = window.copy()
        for channel in (0, 1):
            warped[:, channel, 0] = np.interp(source, frames, window[:, channel, 0])
        if warped[low, 1, 0] < 0:
            copies.append(warped)
    return np.array(copies).reshape(-1, *up.shape[1:])


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


def folds(y: np.ndarray, seed: int) -> np.ndarray:
    """Fold of each video: FOLDS folds with the same share of positives."""
    rng, fold = np.random.default_rng(seed), np.empty(len(y), int)
    for label in (0, 1):
        where = rng.permutation(np.flatnonzero(y == label))
        fold[where] = np.arange(len(where)) % FOLDS
    return fold


def train(x, mask, y, seed: int, epochs: int, flip: bool = True, held_out=(), init=None):
    """Full-batch AdamW on the class-balanced binary cross-entropy, from random weights or with
    the convolutions from `init` (a pretrained encoder). Returns the model and, per epoch, the
    training loss and the loss on each (x, mask, y) in `held_out`, which nothing uses to choose."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = network(x.shape[2] * (x.shape[3] + 1))
    if init is not None:
        model[:ENCODER].load_state_dict(init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    target = torch.tensor(y, dtype=torch.float32)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - y).sum() / y.sum()))
    held_out = [(features(hx, hm), torch.tensor(hy, dtype=torch.float32))
                for hx, hm, hy in held_out]  # fmt: skip
    curve = []
    for _ in range(epochs):
        model.train()
        error = loss(model(features(*augment(x, mask, rng, flip))).squeeze(1), target)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            held = [loss(model(f).squeeze(1), t).item() for f, t in held_out]
        curve.append([error.item(), *held])
    return model.eval(), curve


@torch.no_grad()
def probability(model: nn.Module, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return torch.sigmoid(model(features(x, mask)).squeeze(1)).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--arms", action="store_true", help="train M1b on the arms only")
    parser.add_argument("--heights", action="store_true", help="train M1e on arm heights only")
    parser.add_argument("--init", help="pretrained encoders in experiments/serve, e.g. pretrain")
    parser.add_argument("--more", help="clips of dataset.NEW to train on too, by name: jc-post")
    parser.add_argument("--synthetic", action="store_true", help="M1g: add synthetic faults")
    parser.add_argument("--name", help="of the files in experiments/serve; default m1 or m1b")
    args = parser.parse_args()
    if args.init and not args.arms:
        parser.error("the pretrained encoder takes the arms only: add --arms")
    if args.synthetic and not args.heights:
        parser.error("synthetic faults are made on the arm heights: add --heights")
    data = np.load(dataset.OUT)
    name = args.name or (
        "m2" if args.init else "m1e" if args.heights else "m1b" if args.arms else "m1"
    )
    flip = not (args.arms or args.heights)
    inits = (torch.load(baseline.RESULTS / f"{args.init}.pt")["encoders"] if args.init
             else [None] * args.seeds)  # fmt: skip
    x, mask = data["x"], data["mask"]
    if args.heights or args.arms:
        x, mask = (heights if args.heights else arms)(x, mask)
    labeled = data["valid"] == 1
    train_, test = (labeled & (data["split"] == s) for s in ("train", "test"))
    if args.more:
        train_ |= labeled & (data["split"] == "new") & np.char.startswith(data["video"], args.more)
    truth = data["category"]
    y, y_test = ((truth[s] == baseline.POSITIVE).astype(np.float32) for s in (train_, test))
    xt, mt = x[train_], mask[train_]
    weights = sum(p.numel() for p in network(x.shape[2] * (x.shape[3] + 1)).parameters())
    print(f"{name.upper()}: {weights} weights, {train_.sum()} train videos ({int(y.sum())} drop "
          f"early), {FOLDS}-fold cross-validation for the validation scores")  # fmt: skip

    def fitted(x, mask, y, seed):
        """The training serves, plus their synthetic faults with --synthetic."""
        if not args.synthetic:
            return x, mask, y
        made = synthetic(x[y == 0], np.random.default_rng(seed))
        return (np.concatenate([x, made]), np.concatenate([mask, np.ones(made.shape[:3], bool)]),
                np.concatenate([y, np.ones(len(made), np.float32)]))  # fmt: skip

    rows, curves, results, states = [], [], {s: [] for s in ("train", "validation", "test")}, []
    for seed in range(args.seeds):
        fold, validation, fold_loss = folds(y, seed), np.zeros(len(y)), []
        for k in range(FOLDS):
            fit, out = fold != k, fold == k
            model, curve = train(*fitted(xt[fit], mt[fit], y[fit], seed), seed, args.epochs,
                                 flip, [(xt[out], mt[out], y[out])], inits[seed])  # fmt: skip
            validation[out] = probability(model, xt[out], mt[out])
            fold_loss.append(np.array(curve)[:, 1])
        model, curve = train(*fitted(xt, mt, y, seed), seed, args.epochs, flip,
                             [(x[test], mask[test], y_test)], inits[seed])  # fmt: skip
        states.append(model.state_dict())
        val_loss = np.mean(fold_loss, 0)
        for epoch, ((fit_loss, test_loss), val) in enumerate(zip(curve, val_loss, strict=True), 1):
            curves.append({"model": name, "seed": seed, "epoch": epoch, "train": fit_loss,
                           "validation": val, "test": test_loss})  # fmt: skip
        scores = {"train": (train_, probability(model, xt, mt)), "validation": (train_, validation),
                  "test": (test, probability(model, x[test], mask[test]))}  # fmt: skip
        line = []
        for split, (chosen, score) in scores.items():
            f1 = macro_prf(truth[chosen].tolist(), baseline.predict(score, 0.5))[2]
            area = baseline.auroc(truth[chosen] == baseline.POSITIVE, score)
            results[split].append((f1, area))
            line.append(f"{split} macro-F1 {f1:.3f} AUROC {area:.3f}")
            rows += baseline.scored(name, seed, split, data, chosen, score)
        print(f"  seed {seed}: " + ", ".join(line))
    for split, values in results.items():
        (f1, sd_f1), (area, sd_area) = zip(np.mean(values, 0), np.std(values, 0))
        print(f"  {split}: macro-F1 {f1:.3f} ± {sd_f1:.3f}, AUROC {area:.3f} ± {sd_area:.3f}")
    baseline.save(name, rows)
    baseline.save(f"{name}-curves", curves)
    reference = torch.from_numpy(xt[y == 0].mean(0))  # a clean serve, for strokes.serve.model
    torch.save({"arms": args.arms, "heights": args.heights, "epochs": args.epochs,
                "weights": states, "reference": reference}, baseline.RESULTS / f"{name}.pt")  # fmt: skip
    plot(name, curves)


def plot(name: str, curves: list[dict]) -> None:
    """Mean loss per epoch over seeds, the training curve of the report."""
    _, ax = plt.subplots(figsize=(6, 3.5))
    for split in ("train", "validation", "test"):
        loss = np.array([[c[split] for c in curves if c["seed"] == s]
                         for s in sorted({c["seed"] for c in curves})])  # fmt: skip
        ax.plot(np.arange(1, loss.shape[1] + 1), loss.mean(0), label=split)
    ax.set(xlabel="epoch", ylabel="class-balanced BCE", title=f"{name.upper()}: loss per epoch")
    ax.legend()
    plt.tight_layout()
    plt.savefig(baseline.RESULTS / f"{name}-curves.png", dpi=150)


if __name__ == "__main__":
    main()
