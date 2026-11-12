"""M3, the serve beyond the tossing arm: one output per aspect of the body the team answered in
the "Cuerpo" tab of eval/label.py, and a pose measure per aspect that needs no training.

    uv run python -m strokes.serve.body [--name ernesto] [--seeds 5] [--epochs 85] [--travel]

Every window is first turned so the tossing arm is the left one and x grows toward the court
(players land into it), so a measure reads the same for every player and camera side. Each
measure is signed so that a higher value means more to improve; the sign is fixed here from what
the question asks, not fitted to the answers (eval/body.py scores how well each one follows them).

M3 is M1's network (strokes/serve/train.py) with one output per aspect of ANSWERS ("needs work"),
trained from scratch on the 17 joints, each measured from that frame's hips and with no mask (the
lesson of M1e: against the body, not where it is in the image). The binary cross-entropy is
class-balanced per aspect and skips the aspects a serve has no answer for. Seeds, 85 epochs and
augmentation are M1e's; nothing is tuned. Validation scores come from 5 folds that keep each
recording session on one side (SESSION), over the answered serves except the test split (jc-s28
to s43 and pro01) and the YouTube clips (FRESH), which are only scored. Each aspect's threshold is
the one with the best F1 for "needs work" on validation. Aspects whose retest kappa
(experiments/serve/body.csv, from eval/body.py) is under MIN_KAPPA are left out. Writes
experiments/serve/m3.csv (scores), m3-curves.csv/.png and m3.pt.
--travel trains M3b: M3 with the positions from the hips at contact, as the dataset has them,
instead of from each frame's hips, so the input keeps how far the body moves (the hips going down
before the jump, the step into the court). It was added after M3's checks (eval/body.py) showed
that M3 cannot see that travel; it is scored by the same checks and lines.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn

from core import ROOT
from strokes.serve import baseline, dataset, train

J = dataset.JOINTS.index
LOAD = 30  # frames before contact the legs bend in: 1 s
NEAR = 3  # frames on each side of contact for the reach: 0.1 s, the contact frame is found early
ANSWERED = ROOT / "eval" / "serve"  # aspects_<name>.csv
SESSION = re.compile(r"[-_]s\d+$")  # jc-post2-s07 is session jc-post2, pro02_s01 is pro02
FRESH = ("pro05", "yt")  # the YouTube clips of Nov 9: scored, never trained on
MIN_KAPPA = 0.40  # agreement of an annotator with themself under which an aspect is left out
MIN_SEPARATES = 0.80  # amateurs-against-pros AUROC from which a measure is shown next to the pros'
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


def toss_height(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Height of the tossing wrist above its shoulder when the racket wrist is lowest, per window:
    the labeling question as a number (eval/improvement.py)."""
    up = train.heights(x, mask)[0][..., 0]  # tossing elbow, wrist, racket elbow, wrist
    look = dataset.BEFORE - train.LOOK
    low = look + up[:, look : dataset.BEFORE, 3].argmin(1)
    return up[np.arange(len(up)), low, 1]


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


def relative(x: np.ndarray, travel: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """oriented() windows with each joint relative to that frame's hips (or, with `travel`, to
    the hips at contact, as they are), and all points seen."""
    if not travel:
        x = x - x[:, :, dataset.HIPS].mean(2, keepdims=True)
    return x, np.ones(x.shape[:3], bool)


def inputs(x: np.ndarray, mask: np.ndarray, travel: bool = False):
    """M3's (or M3b's) input from dataset windows."""
    return relative(oriented(x, mask)[0], travel)


def session_folds(videos, seed: int) -> np.ndarray:
    """Fold of each serve: whole sessions, the largest first, each into the fold with the fewest
    serves so far (ties in a random order of the seed)."""
    sessions = np.array([SESSION.sub("", str(v)) for v in videos])
    names, sizes = np.unique(sessions, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(names))
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold, filled = {}, np.zeros(train.FOLDS)
    for i in order:
        fold[names[i]] = int(filled.argmin())
        filled[fold[names[i]]] += sizes[i]
    return np.array([fold[s] for s in sessions])


def fit(x, mask, y, seed: int, epochs: int, held_out=()):
    """train.train with one output per aspect (y: NaN where a serve has no answer). Returns the
    model and, per epoch, the training loss and the loss on each (x, mask, y) in `held_out`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = train.network(x.shape[2] * (x.shape[3] + 1), outputs=y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    positives = np.nansum(y, 0)
    balance = (~np.isnan(y)).sum(0) - positives
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(
        balance / np.maximum(positives, 1), dtype=torch.float32))  # fmt: skip

    def loss(x, mask, y):
        seen = torch.from_numpy(~np.isnan(y))
        error = criterion(model(train.features(x, mask)), torch.from_numpy(np.nan_to_num(y)))
        return (error * seen).sum() / seen.sum()

    curve = []
    for _ in range(epochs):
        model.train()
        error = loss(*train.augment(x, mask, rng, flip=False), y)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            curve.append([error.item(), *(loss(*held).item() for held in held_out)])
    return model.eval(), curve


@torch.no_grad()
def probability(model: nn.Module, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N, aspects) probability of "needs work"."""
    return torch.sigmoid(model(train.features(x, mask))).numpy()


def load(name: str = "m3") -> dict:
    """Saved M3 or M3b: its aspects, thresholds, reference serve and one network per seed."""
    saved = {"name": name, "travel": False, **torch.load(baseline.RESULTS / f"{name}.pt")}
    networks = []
    for state in saved["weights"]:
        networks.append(train.network(outputs=len(saved["aspects"])))
        networks[-1].load_state_dict(state)
        networks[-1].eval()
    return {**saved, "networks": networks}


def predict(m3: dict, x: np.ndarray, mask: np.ndarray, prepared: bool = False) -> np.ndarray:
    """(N, aspects) probability of "needs work", mean over seeds, of dataset windows (or of M3's
    inputs when `prepared`)."""
    if not prepared:
        x, mask = inputs(x, mask, m3["travel"])
    return np.mean([probability(network, x, mask) for network in m3["networks"]], 0)


def threshold(target: np.ndarray, score: np.ndarray) -> float:
    """The probability at or above which "needs work" has the best F1 on the answered serves."""
    have = ~np.isnan(target)
    truth, score = target[have] == 1, score[have]
    best, cut = -1.0, 0.5
    for candidate in np.unique(score):
        said = score >= candidate
        f1 = 2 * (said & truth).sum() / max(said.sum() + truth.sum(), 1)
        if f1 > best:
            best, cut = f1, float(candidate)
    return cut


def kept() -> list[str]:
    """The aspects of ANSWERS whose retest kappa, once there is one, is MIN_KAPPA or more."""
    path, low = baseline.RESULTS / "body.csv", set()
    if path.exists():
        with open(path, newline="") as file:
            low = {r["aspect"] for r in csv.DictReader(file)
                   if r["retest_kappa"] not in ("", "nan") and float(r["retest_kappa"]) < MIN_KAPPA}  # fmt: skip
    return [aspect for aspect in ANSWERS if aspect not in low]


def splits(data, y: np.ndarray) -> dict[str, np.ndarray]:
    """The serves M3 is trained and validated on (pool), its test split and the fresh clips."""
    judged = data["valid"] == 1
    fresh = judged & np.array([str(v).startswith(FRESH) for v in data["video"]])
    test = judged & ~np.isnan(y).all(1) & (data["split"] == "test")
    return {"pool": judged & ~np.isnan(y).all(1) & ~fresh & ~test, "test": test, "fresh": fresh}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="ernesto", help="whose answers: aspects_<name>.csv")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=85)
    parser.add_argument("--travel", action="store_true", help="M3b: keep how far the body moves")
    args = parser.parse_args()
    name = "m3b" if args.travel else "m3"
    data = np.load(dataset.OUT)
    aspects = kept()
    y = targets(answers(ANSWERED / f"aspects_{args.name}.csv"), data["video"])
    y = y[:, [list(ANSWERS).index(a) for a in aspects]]
    chosen = splits(data, y)
    pool, test, fresh = chosen["pool"], chosen["test"], chosen["fresh"]
    x, mask = inputs(data["x"], data["mask"], args.travel)
    xp, mp, yp = x[pool], mask[pool], y[pool]
    print(f"{name.upper()}: {len(aspects)} aspects ({', '.join(aspects)}), {pool.sum()} serves to train and "
          f"validate on, {test.sum()} test serves, {fresh.sum()} fresh")  # fmt: skip
    rows, curves, states, validations = [], [], [], []
    for seed in range(args.seeds):
        fold, fold_loss = session_folds(data["video"][pool], seed), []
        validation = np.zeros_like(yp)
        for k in range(train.FOLDS):
            fit_, out = fold != k, fold == k
            model, curve = fit(xp[fit_], mp[fit_], yp[fit_], seed, args.epochs,
                               [(xp[out], mp[out], yp[out])])  # fmt: skip
            validation[out] = probability(model, xp[out], mp[out])
            fold_loss.append(np.array(curve)[:, 1])
        model, curve = fit(xp, mp, yp, seed, args.epochs, [(x[test], mask[test], y[test])])
        states.append(model.state_dict())
        validations.append(validation)
        for epoch, ((fit_loss, test_loss), val) in enumerate(
            zip(curve, np.mean(fold_loss, 0), strict=True), 1
        ):
            curves.append({"model": name, "seed": seed, "epoch": epoch, "train": fit_loss,
                           "validation": val, "test": test_loss})  # fmt: skip
        scores = {"train": (pool, probability(model, xp, mp)), "validation": (pool, validation),
                  "test": (test, probability(model, x[test], mask[test])),
                  "fresh": (fresh, probability(model, x[fresh], mask[fresh]))}  # fmt: skip
        for split, (where, score) in scores.items():
            for v, player, values in zip(data["video"][where], data["player"][where], score):
                rows += [{"model": name, "seed": seed, "split": split, "video": v, "player": player,
                          "aspect": aspect, "probability": round(float(p), 5)}
                         for aspect, p in zip(aspects, values)]  # fmt: skip
    mean = np.mean(validations, 0)
    cuts = {aspect: threshold(yp[:, k], mean[:, k]) for k, aspect in enumerate(aspects)}
    for k, aspect in enumerate(aspects):
        have = ~np.isnan(yp[:, k])
        areas = [baseline.auroc(yp[have, k] == 1, v[have, k]) for v in validations]
        print(f"  {aspect}: validation AUROC {np.mean(areas):.3f} ± {np.std(areas):.3f} "
              f"({int(np.nansum(yp[:, k]))} of {have.sum()} need work), threshold {cuts[aspect]:.3f}")  # fmt: skip
    baseline.save(name, rows)
    baseline.save(f"{name}-curves", curves)
    train.plot(name, curves)
    fine = (np.nan_to_num(yp) == 0).all(1)  # a serve with nothing to improve, for occlusion maps
    torch.save({"aspects": aspects, "epochs": args.epochs, "weights": states, "thresholds": cuts,
                "travel": args.travel, "reference": torch.from_numpy(xp[fine].mean(0))},
               baseline.RESULTS / f"{name}.pt")  # fmt: skip


if __name__ == "__main__":
    main()
