"""M3, the serve beyond the tossing arm: one output per aspect of the body the team answered in
the "Cuerpo" tab of eval/label.py, and a pose measure per aspect that needs no training.

    uv run python -m strokes.serve.body [--name ernesto] [--seeds 5] [--epochs 85] [--travel | --balanced | --parts]

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
--balanced trains M3c and --parts M4, both written after M3b's checks and against what they showed:
the answers follow who serves (the racket drop needs work on 49 of EA's 54 serves and 7 of JC's
81), so a network that tells the players apart already scores well. M3c is M3b with two changes.
Its loss weighs every (player, answer) cell of an aspect the same (cell_weights), so telling the
players apart no longer lowers it. And its windows are shifted in time so that contact is where
the racket wrist is highest (oriented with at_reach): dataset.contact finds contact about 0.1 s
late when the tossing arm is still up while the racket goes up, on 53 of EA's 54 serves and 15 of
JC's 81, which alone tells EA from JC. M4 is M3c with one network per part of the body (HEADS),
each seeing only that part (prepared), the lesson of M1e again.
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
MIN_CELL = 5  # serves under which a (player, answer) cell weighs as if it had this many
LOWER = list(range(dataset.HIPS[0], len(dataset.JOINTS)))  # hips to feet
HEADS = {  # M4: the aspects each network answers; it sees only that part of the body (prepared)
    "lower": ("legs", "landing"),
    "racket": ("racket_at_peak", "racket_drop", "arm_at_contact"),
}
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
# What an audit of the measured instants (Nov 12, stick figures of 12 serves and the frames of two)
# changed, once and without looking at the answers. Every measure is read with contact where the
# racket wrist is highest (oriented with at_reach). And the racket wrist of JC and EA never dips
# after the trophy, so "its lowest" was the trophy frame itself, where the racket is still coming
# up: the racket drop is now how far that wrist falls, and the time of that lowest point, which
# stood for the racket at the ball's peak, is left out (the pose does not show the ball).
AUDITED = {
    "racket_at_peak": "none: the pose does not show where the racket is at the ball's peak",
    "racket_drop": "largest fall of the racket wrist between the trophy and contact, negated",
}


def reach(x: np.ndarray) -> np.ndarray:
    """Frame of each oriented() window where the racket wrist is highest above its shoulder, from
    the trophy (the tossing wrist highest) on: where the ball is hit."""
    y = x[..., 1]  # image y grows downward
    before = dataset.BEFORE
    trophy = (y[:, :before, J("left_shoulder")] - y[:, :before, J("left_wrist")]).argmax(1)
    racket = y[:, :, J("right_shoulder")] - y[:, :, J("right_wrist")]
    return np.where(np.arange(x.shape[1])[None] >= trophy[:, None], racket, -np.inf).argmax(1)


def oriented(x: np.ndarray, mask: np.ndarray, at_reach: bool = False):
    """Copies of the windows with the tossing arm on the left and x toward the court. With
    `at_reach` each window is also shifted in time so that contact (frame dataset.BEFORE) is at
    reach(), with the hips at that frame at the origin; the frames the shift leaves empty repeat
    the first or the last one."""
    x, mask = x.copy(), mask.copy()
    train.mirror(x, mask, np.array([baseline.tossing_side(w) == "right" for w in x], bool))
    hips = x[:, :, dataset.HIPS, 0].mean(2)
    x[..., 0] *= np.where(hips[:, -1] >= hips[:, 0], 1, -1)[:, None, None]
    if at_reach:
        frames = np.arange(x.shape[1])[None] + (reach(x) - dataset.BEFORE)[:, None]
        frames, rows = np.clip(frames, 0, x.shape[1] - 1), np.arange(len(x))[:, None]
        x, mask = x[rows, frames], mask[rows, frames]
        x -= x[:, dataset.BEFORE, dataset.HIPS].mean(1)[:, None, None]
    return x, mask


def instants(x: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Frame of each window, as dataset.window cut it, of two instants the audited measures read:
    contact (reach) and, in the second before it, the hips at their lowest (the knees most bent)."""
    o, _ = oriented(x, mask)
    contact, frames = reach(o), np.arange(x.shape[1])[None]
    loading = (frames >= contact[:, None] - LOAD) & (frames < contact[:, None])
    hips = o[:, :, dataset.HIPS, 1].mean(2)  # image y grows downward
    return {"bend": np.where(loading, hips, -np.inf).argmax(1), "contact": contact}


def toss_height(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Height of the tossing wrist above its shoulder when the racket wrist is lowest, per window:
    the labeling question as a number (eval/improvement.py)."""
    up = train.heights(x, mask)[0][..., 0]  # tossing elbow, wrist, racket elbow, wrist
    look = dataset.BEFORE - train.LOOK
    low = look + up[:, look : dataset.BEFORE, 3].argmin(1)
    return up[np.arange(len(up)), low, 1]


def measures(x: np.ndarray, mask: np.ndarray, audited: bool = False) -> dict[str, np.ndarray]:
    """One value per window and aspect of MEASURES, higher meaning more to improve; with
    `audited`, as the audit left them (AUDITED), NaN where there is no measure."""
    x, _ = oriented(x, mask, audited)
    y = x[..., 1]  # image y grows downward
    before, rows = dataset.BEFORE, np.arange(len(x))
    hips = x[:, :, dataset.HIPS].mean(2)
    racket = y[:, :, J("right_shoulder")] - y[:, :, J("right_wrist")]  # above its shoulder
    trophy = (y[:, :before, J("left_shoulder")] - y[:, :before, J("left_wrist")]).argmax(1)
    # The racket drops behind the back between the trophy (tossing wrist highest) and contact
    after_trophy = np.arange(before)[None] >= trophy[:, None]
    low = np.where(after_trophy, racket[:, :before], np.inf).argmin(1)
    tilt = y[:, :, J("right_shoulder")] - y[:, :, J("left_shoulder")]
    values = {
        "legs": np.median(hips[:, :10, 1], 1) - hips[:, before - LOAD : before, 1].max(1),
        "shoulders": -tilt[rows, trophy],
        "racket_at_peak": (before - low) / dataset.FPS,
        "racket_drop": racket[rows, low],
        "arm_at_contact": -racket[:, before - NEAR : before + NEAR + 1].max(1),
        "landing": hips[:, before, 0] - hips[:, -1, 0],
    }
    if audited:
        swing = np.arange(before + 1)[None] >= trophy[:, None]  # from the trophy to contact
        top = np.maximum.accumulate(np.where(swing, racket[:, : before + 1], -np.inf), 1)
        values["racket_drop"] = -np.where(swing, top - racket[:, : before + 1], 0).max(1)
        values["racket_at_peak"] = np.full(len(x), np.nan)
    return values


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


def prepared(o: np.ndarray, travel: bool = False, parts: bool = False) -> dict[str, tuple]:
    """The (x, mask) input of each network from oriented() windows: one network on the whole body
    (relative), or with `parts` one per HEADS that sees only its part: the joints from the hips to
    the feet, from the hips at contact; and train.heights' elbow and wrist of each arm above its
    own shoulder. Every point counts as seen."""
    if not parts:
        return {"body": relative(o, travel)}
    y = o[:, :, train.ARMS, 1]  # image y grows downward
    up = (y[:, :, [0, 0, 3, 3]] - y[:, :, [1, 2, 4, 5]])[..., None]
    lower = o[:, :, LOWER]
    return {"lower": (lower, np.ones(lower.shape[:3], bool)),
            "racket": (up, np.ones(up.shape[:3], bool))}  # fmt: skip


def heads(aspects: list[str], parts: bool = False) -> dict[str, list[int]]:
    """The columns of `aspects` each network answers."""
    if not parts:
        return {"body": list(range(len(aspects)))}
    missing = set(aspects) - {a for mine in HEADS.values() for a in mine}
    if missing:
        raise ValueError(f"no part of the body in HEADS for {sorted(missing)}")
    columns = {h: [aspects.index(a) for a in mine if a in aspects] for h, mine in HEADS.items()}
    return {head: mine for head, mine in columns.items() if mine}


def cell_weights(y: np.ndarray, player: np.ndarray) -> np.ndarray:
    """(N, aspects) weight of each answer so that, per aspect, every (player, answer) cell weighs
    the same: telling the players apart no longer lowers the loss, only what changes within one
    does. The professionals count as one player. A cell under MIN_CELL serves weighs as if it had
    MIN_CELL, so that one serve never stands for a whole cell."""
    player = player.astype(str)
    group = np.where(np.char.startswith(player, "pro"), "pro", player)
    weight = np.zeros(y.shape, np.float32)
    for k in range(y.shape[1]):
        for g in np.unique(group):
            for answer in (0, 1):
                cell = (group == g) & (y[:, k] == answer)
                if cell.any():
                    weight[cell, k] = 1 / max(cell.sum(), MIN_CELL)
    return weight


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


def fit(x, mask, y, seed: int, epochs: int, held_out=(), weight=None):
    """train.train with one output per aspect (y: NaN where a serve has no answer). The loss is
    class-balanced per aspect, or with `weight` (cell_weights) weighs each answer by it. Returns
    the model and, per epoch, the training loss and the loss on each (x, mask, y) in `held_out`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = train.network(x.shape[2] * (x.shape[3] + 1), outputs=y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    positives = np.nansum(y, 0)
    balance = (~np.isnan(y)).sum(0) - positives
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=None if weight is not None
                                     else torch.tensor(balance / np.maximum(positives, 1),
                                                       dtype=torch.float32))  # fmt: skip

    def loss(x, mask, y, weight=None):
        seen = torch.from_numpy(~np.isnan(y) if weight is None else weight)
        error = criterion(model(train.features(x, mask)), torch.from_numpy(np.nan_to_num(y)))
        return (error * seen).sum() / seen.sum()

    curve = []
    for _ in range(epochs):
        model.train()
        error = loss(*train.augment(x, mask, rng, flip=False), y, weight)
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


def fit_heads(o, y, seed: int, epochs: int, spec: dict, player=None, held_out=None):
    """fit() for each network of `spec` (its aspects and whether travel, parts and balanced), on
    oriented() windows. `held_out` is an (o, y) pair. Returns the networks by head and, per
    epoch, the training and held-out loss, mean over the networks."""
    inputs = prepared(o, spec["travel"], spec["parts"])
    columns = heads(spec["aspects"], spec["parts"])
    held = prepared(held_out[0], spec["travel"], spec["parts"]) if held_out else None
    weight = cell_weights(y, player) if spec["balanced"] else None
    networks, curves = {}, []
    for head, mine in columns.items():
        out = [(*held[head], held_out[1][:, mine])] if held_out else []
        networks[head], curve = fit(*inputs[head], y[:, mine], seed, epochs, out,
                                    None if weight is None else weight[:, mine])  # fmt: skip
        curves.append(curve)
    return networks, np.mean(curves, 0)


def probabilities(networks: dict, o: np.ndarray, spec: dict) -> np.ndarray:
    """(N, aspects) probability of "needs work" of oriented() windows, by one seed's networks."""
    inputs = prepared(o, spec["travel"], spec["parts"])
    out = np.zeros((len(o), len(spec["aspects"])), np.float32)
    for head, mine in heads(spec["aspects"], spec["parts"]).items():
        out[:, mine] = probability(networks[head], *inputs[head])
    return out


def load(name: str = "m3") -> dict:
    """A saved body model: its aspects, thresholds, how its input is made (travel, at_reach,
    parts) and, per seed, its networks by head. M3 and M3b were saved as one network per seed."""
    saved = {"name": name, "travel": False, "at_reach": False, "balanced": False, "parts": False,
             **torch.load(baseline.RESULTS / f"{name}.pt")}  # fmt: skip
    states = saved.get("heads") or {"body": saved["weights"]}  # head: one state per seed
    networks = []
    for seed in range(len(next(iter(states.values())))):
        networks.append({})
        for head, mine in heads(saved["aspects"], saved["parts"]).items():
            state = states[head][seed]
            networks[-1][head] = train.network(state["0.weight"].shape[1], outputs=len(mine))
            networks[-1][head].load_state_dict(state)
            networks[-1][head].eval()
    return {**saved, "networks": networks}


def predict_oriented(m3: dict, o: np.ndarray) -> np.ndarray:
    """(N, aspects) probability of "needs work", mean over seeds, of oriented() windows."""
    return np.mean([probabilities(networks, o, m3) for networks in m3["networks"]], 0)


def predict(m3: dict, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N, aspects) probability of "needs work", mean over seeds, of dataset windows."""
    return predict_oriented(m3, oriented(x, mask, m3["at_reach"])[0])


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
    parser.add_argument("--balanced", action="store_true",
                        help="M3c: M3b with contact at the reach and every (player, answer) cell "
                             "weighing the same")  # fmt: skip
    parser.add_argument("--parts", action="store_true",
                        help="M4: M3c with one network per part of the body")  # fmt: skip
    args = parser.parse_args()
    balanced = args.balanced or args.parts
    name = "m4" if args.parts else "m3c" if balanced else "m3b" if args.travel else "m3"
    data = np.load(dataset.OUT)
    aspects = kept()
    spec = {"aspects": aspects, "travel": args.travel or balanced, "at_reach": balanced,
            "balanced": balanced, "parts": args.parts}  # fmt: skip
    y = targets(answers(ANSWERED / f"aspects_{args.name}.csv"), data["video"])
    y = y[:, [list(ANSWERS).index(a) for a in aspects]]
    chosen = splits(data, y)
    pool, test, fresh = chosen["pool"], chosen["test"], chosen["fresh"]
    o = oriented(data["x"], data["mask"], spec["at_reach"])[0]
    op, yp, players = o[pool], y[pool], data["player"][pool]
    print(f"{name.upper()}: {len(aspects)} aspects ({', '.join(aspects)}), {pool.sum()} serves to train and "
          f"validate on, {test.sum()} test serves, {fresh.sum()} fresh")  # fmt: skip
    rows, curves, states, validations = [], [], [], []
    for seed in range(args.seeds):
        fold, fold_loss = session_folds(data["video"][pool], seed), []
        validation = np.zeros_like(yp)
        for k in range(train.FOLDS):
            fit_, out = fold != k, fold == k
            networks, curve = fit_heads(op[fit_], yp[fit_], seed, args.epochs, spec,
                                        players[fit_], (op[out], yp[out]))  # fmt: skip
            validation[out] = probabilities(networks, op[out], spec)
            fold_loss.append(curve[:, 1])
        networks, curve = fit_heads(op, yp, seed, args.epochs, spec, players, (o[test], y[test]))
        states.append({head: network.state_dict() for head, network in networks.items()})
        validations.append(validation)
        for epoch, ((fit_loss, test_loss), val) in enumerate(
            zip(curve, np.mean(fold_loss, 0), strict=True), 1
        ):
            curves.append({"model": name, "seed": seed, "epoch": epoch, "train": fit_loss,
                           "validation": val, "test": test_loss})  # fmt: skip
        scores = {"train": (pool, probabilities(networks, op, spec)),
                  "validation": (pool, validation),
                  "test": (test, probabilities(networks, o[test], spec)),
                  "fresh": (fresh, probabilities(networks, o[fresh], spec))}  # fmt: skip
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
    saved = {**spec, "epochs": args.epochs, "thresholds": cuts,
             "heads": {head: [state[head] for state in states] for head in states[0]}}  # fmt: skip
    if not args.parts:  # a serve with nothing to improve, for occlusion maps
        fine = (np.nan_to_num(yp) == 0).all(1)
        saved["reference"] = torch.from_numpy(prepared(op[fine], spec["travel"])["body"][0].mean(0))
    torch.save(saved, baseline.RESULTS / f"{name}.pt")


if __name__ == "__main__":
    main()
