"""How far the team's answers about the body can be trusted, with no second annotator.

    uv run python -m eval.body [--name ernesto] [--checks [--model m3b]]

Reads data/processed/serve_dataset.npz (python -m strokes.serve.dataset) and the answers of the
"Cuerpo" tab, eval/serve/aspects_<name>.csv. For each aspect of strokes/serve/body.ANSWERS:

1. Answer against measure: AUROC of the pose measure (strokes/serve/body.MEASURES, higher means
   more to improve, with its direction fixed beforehand) for the answer "needs work", on every
   answered serve and within EA and within JC alone, where each class has MIN_CLASS serves or
   more. Within a player it says whether the answer follows the serve and not who serves.
2. The professionals: the annotator called every professional serve "as it should be". The share
   of professional serves the measure puts at or past the amateurs' answered as needing work
   (their median) says whether that answer follows the serve or the player.
3. The annotator against themself: once "Cuerpo · repetir" has answers
   (aspects_<name>_retest.csv), Cohen's kappa between both passes on the needs-work answer, and
   on the raw answer. An aspect with a kappa under MIN_KAPPA is not used to train or shown.

Writes experiments/serve/body.csv.

--checks scores M3 (python -m strokes.serve.body), or M3b with --model m3b, each aspect on its
own, and writes experiments/serve/<model>-checks.csv, -occlusion.png and -calibration.png:

1. Within each player: validation AUROC within EA and within JC, against the measure's on the
   same serves.
2. A new player: M3 retrained without EA, scored on EA; and without JC, scored on JC.
3. The professionals (pro01 to pro05, Federer included): share called "needs work".
4. Counterfactual: on serves answered "as it should be", the part of the body the aspect asks
   about is moved toward "needs work" (edited); its probability should rise, and should barely
   move when an unrelated part is moved instead (UNRELATED).
5. Occlusion: the share of the probability that rests on the aspect's own part (OWN, PARTS).
6. Robustness: a mirrored window, 10 to 40 % of the points unseen, the camera in front (THETIS).
7. Calibration: Brier score and reliability curve on validation.
8. Fresh serves: the test split and the YouTube clips of Nov 9 (body.FRESH).

An aspect is adopted, that is shown as M3's answer in the app, only if it passes every line fixed
before M3 was trained: retest kappa MIN_KAPPA or more; within EA and within JC (where both
classes have MIN_CLASS serves) AUROC MIN_WITHIN or more and no lower than its measure's; on the
player left out, AUROC MIN_NEW_PLAYER or more; at most MAX_PROS of the professionals called
"needs work"; and its own edit raising the probability on MIN_RAISED of the serves while the
unrelated edit moves it less than MAX_MOVED on average. Checks 5 to 8 are reported, not lines.
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import ROOT
from eval import bias
from eval.run_eval import kappa
from strokes.serve import baseline, body, dataset, model, pretrain, train

EVAL = ROOT / "eval" / "serve"
MIN_CLASS = 5
MIN_KAPPA = body.MIN_KAPPA
MIN_WITHIN, MIN_NEW_PLAYER, MAX_PROS, MIN_RAISED, MAX_MOVED = 0.70, 0.65, 0.20, 0.80, 0.05
J = dataset.JOINTS.index
PARTS = {  # after body.oriented the tossing arm is the left one
    "head and shoulders": ["nose", "left_shoulder", "right_shoulder"],
    "tossing arm": ["left_elbow", "left_wrist"],
    "racket arm": ["right_elbow", "right_wrist"],
    "legs": list(dataset.JOINTS[7:]),  # hips to feet
}
OWN = {"legs": "legs", "shoulders": "head and shoulders", "racket_at_peak": "racket arm",
       "racket_drop": "racket arm", "arm_at_contact": "racket arm", "landing": "legs"}  # fmt: skip
UNRELATED = {"legs": "toss", "shoulders": "toss", "landing": "toss", "racket_at_peak": "legs",
             "racket_drop": "legs", "arm_at_contact": "legs"}  # fmt: skip
EARLY = 6  # frames the racket arm moves early in the racket_at_peak edit: 0.2 s


def within(positive: np.ndarray, score: np.ndarray, chosen: np.ndarray) -> float:
    """AUROC among the chosen serves, NaN when a class has fewer than MIN_CLASS."""
    have = chosen & ~np.isnan(positive)
    pos = positive[have] == 1
    if min(pos.sum(), (~pos).sum()) < MIN_CLASS:
        return float("nan")
    return baseline.auroc(pos, score[have])


def retest(name: str) -> dict[str, tuple[float, float, int]]:
    """Per aspect: kappa of the needs-work answer and of the raw answer, and serves in both."""
    path = EVAL / f"aspects_{name}_retest.csv"
    if not path.exists():
        return {}
    first, second = body.answers(EVAL / f"aspects_{name}.csv"), body.answers(path)
    both = sorted(set(first) & set(second))
    out = {}
    for aspect, (needs, fine) in body.ANSWERS.items():
        raw = [(first[v][aspect], second[v][aspect]) for v in both]
        kept = needs | fine
        binary = [(a in needs, b in needs) for a, b in raw if a in kept and b in kept]
        out[aspect] = (kappa(*map(list, zip(*binary))) if binary else float("nan"),
                       kappa(*map(list, zip(*raw))) if raw else float("nan"), len(both))  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="ernesto", help="whose answers: aspects_<name>.csv")
    parser.add_argument("--checks", action="store_true", help="score M3 aspect by aspect")
    parser.add_argument("--model", default="m3", help="m3 or m3b")
    args = parser.parse_args()
    data = np.load(dataset.OUT)
    judged = data["valid"] == 1
    videos, player = data["video"][judged], data["player"][judged]
    y = body.targets(body.answers(EVAL / f"aspects_{args.name}.csv"), videos)
    values = body.measures(data["x"][judged], data["mask"][judged])
    pros, amateurs = np.char.startswith(player, "pro"), ~np.char.startswith(player, "pro")
    kappas = retest(args.name)
    rows = []
    print(f"{args.name}'s answers against the pose measures, {(~np.isnan(y)).any(1).sum()} serves")
    for k, (aspect, score) in enumerate(values.items()):
        target = y[:, k]
        answered = ~np.isnan(target)
        needs = answered & (target == 1)
        row = {
            "aspect": aspect, "measure": body.MEASURES[aspect], "answered": int(answered.sum()),
            "needs_work": int(needs.sum()),
            "auroc": within(target, score, answered),
            "auroc_ea": within(target, score, player == "ea"),
            "auroc_jc": within(target, score, player == "jc"),
            "median_needs_work": float(np.median(score[needs & amateurs])),
            "median_fine": float(np.median(score[answered & (target == 0) & amateurs])),
            "median_pros": float(np.median(score[pros])),
            "pros_like_needs_work": float(
                (score[pros] >= np.median(score[needs & amateurs])).mean()
            ),
            "retest_kappa": kappas.get(aspect, (np.nan,) * 3)[0],
            "retest_kappa_raw": kappas.get(aspect, (np.nan,) * 3)[1],
            "retest_serves": kappas.get(aspect, (0,) * 3)[2],
        }  # fmt: skip
        rows.append(row)
        print(f"  {aspect}: {row['needs_work']} of {row['answered']} need work; AUROC "
              f"{row['auroc']:.2f}, within EA {row['auroc_ea']:.2f}, within JC "
              f"{row['auroc_jc']:.2f}; medians needs work {row['median_needs_work']:+.2f}, fine "
              f"{row['median_fine']:+.2f}, pros {row['median_pros']:+.2f} "
              f"({row['pros_like_needs_work']:.0%} of the pros as far as the needs-work median)")  # fmt: skip
        if aspect in kappas:
            print(f"    retest on {row['retest_serves']} serves: kappa {row['retest_kappa']:.2f} "
                  f"(raw answer {row['retest_kappa_raw']:.2f})"
                  + (" under the line, not used" if row["retest_kappa"] < MIN_KAPPA else ""))  # fmt: skip
    baseline.save("body", rows)
    if args.checks:
        checks(data, body.targets(body.answers(EVAL / f"aspects_{args.name}.csv"),
                                  data["video"]), kappas, args.model)  # fmt: skip


def edited(part: str, x: np.ndarray) -> np.ndarray:
    """Copies of body.oriented() windows with one part of the serve moved toward "needs work"."""
    x, before, frames = x.copy(), dataset.BEFORE, np.arange(x.shape[1])
    y = x[..., 1]  # a view; image y grows downward
    trophy = (y[:, :before, J("left_shoulder")] - y[:, :before, J("left_wrist")]).argmax(1)
    racket = [J("right_elbow"), J("right_wrist")]
    if part == "legs":  # no knee bend: above the knees the body stays at its stance height
        hips = y[:, :, dataset.HIPS].mean(2)
        lift = np.clip(hips - np.median(hips[:, :10], 1, keepdims=True), 0, None)
        lift[:, before:] = 0
        upper = [J(j) for j in dataset.JOINTS[:9]]  # nose to hips
        y[:, :, upper] -= lift[..., None]
        y[:, :, [J("left_knee"), J("right_knee")]] -= lift[..., None] / 2
    elif part == "shoulders":  # level shoulders until contact
        both = [J("left_shoulder"), J("right_shoulder")]
        y[:, :before, both] = y[:, :before, both].mean(2, keepdims=True)
    elif part == "racket_at_peak":  # the racket arm EARLY frames ahead from the trophy on
        for i, start in enumerate(trophy):
            moved = np.arange(start, before)
            x[i, moved[:, None], racket] = x[i, np.minimum(moved + EARLY, before)[:, None], racket]
    elif part == "racket_drop":  # the racket hand never goes below its shoulder after the trophy
        for i, start in enumerate(trophy):
            shoulder = y[i, start:before, J("right_shoulder")]
            for j in racket:
                y[i, start:before, j] = np.minimum(y[i, start:before, j], shoulder)
    elif part == "arm_at_contact":  # the racket arm lower and bent at contact
        weight = np.clip(1 - np.abs(frames - before) / 6, 0, None)
        y[:, :, J("right_wrist")] += 0.4 * weight
        y[:, :, J("right_elbow")] += 0.2 * weight
    elif part == "landing":  # no step into the court after contact
        hips = x[:, :, dataset.HIPS, 0].mean(2)
        x[:, before:, :, 0] -= (hips[:, before:] - hips[:, before : before + 1])[..., None]
    elif part == "toss":  # the tossing arm 0.5 torso lengths lower in the 0.6 s before contact
        weight = np.clip((frames - before + train.LOOK) / train.LOOK, 0, 1)
        y[:, :, [J("left_elbow"), J("left_wrist")]] += 0.5 * weight[None, :, None]
    else:
        raise ValueError(part)
    return x


def oos(model_name: str, names) -> dict[str, np.ndarray]:
    """M3's scores the serves were not trained on (validation, test or fresh), mean over seeds,
    by video: one value per aspect of `names`."""
    sums = {}
    with open(baseline.RESULTS / f"{model_name}.csv", newline="") as file:
        for r in csv.DictReader(file):
            if r["split"] != "train":
                values = sums.setdefault(r["video"], {}).setdefault(r["aspect"], [])
                values.append(float(r["probability"]))
    return {v: np.array([np.mean(a[n]) for n in names]) for v, a in sums.items()}


def new_player(data, x, mask, y, chosen: dict, player: str, m3: dict):
    """The serves of `player` and M3's scores for them, trained again without that player."""
    fit_ = chosen["pool"] & (data["player"] != player)
    held = (chosen["pool"] | chosen["test"]) & (data["player"] == player)
    scores = []
    for seed in range(len(m3["weights"])):
        network, _ = body.fit(x[fit_], mask[fit_], y[fit_], seed, m3["epochs"])
        scores.append(body.probability(network, x[held], mask[held]))
    return held, np.mean(scores, 0)


def checks(data, y_all: np.ndarray, kappas: dict, model_name: str) -> None:
    m3 = body.load(model_name)
    names = m3["aspects"]
    y = y_all[:, [list(body.ANSWERS).index(a) for a in names]]
    chosen = body.splits(data, y)
    pool, test, fresh = chosen["pool"], chosen["test"], chosen["fresh"]
    player = data["player"]
    scored = oos(model_name, names)
    score = np.array([scored.get(v, np.full(len(names), np.nan)) for v in data["video"]])
    values = body.measures(data["x"], data["mask"])
    x, mask = body.inputs(data["x"], data["mask"], m3["travel"])
    o = body.oriented(data["x"], data["mask"])[0]
    cuts = np.array([m3["thresholds"][a] for a in names])
    rows = []

    def add(aspect, check, value, line=None, passes=None, note=""):
        rows.append({"aspect": aspect, "check": check, "value": round(float(value), 3),
                     "line": "" if line is None else line,
                     "passes": "" if passes is None else "yes" if passes else "no", "note": note})  # fmt: skip

    left_out = {p: new_player(data, x, mask, y, chosen, p, m3) for p in ("ea", "jc")}
    judged = pool | test | fresh
    pros = judged & np.char.startswith(player, "pro")
    occluded = occlusion(m3, x[pool | test], score)
    plot_occlusion(occluded, y[pool | test], names, model_name)
    for k, aspect in enumerate(names):
        target = y[:, k]
        # 0. The annotator against themself
        found = aspect in kappas and not np.isnan(kappas[aspect][0])
        add(aspect, "0 retest kappa", kappas[aspect][0] if found else np.nan, MIN_KAPPA,
            kappas[aspect][0] >= MIN_KAPPA if found else None,
            "" if found else "pending: Cuerpo · repetir")  # fmt: skip
        # 1. Within each player, against the measure
        verdicts = []
        for who in ("ea", "jc"):
            mine = within(target, score[:, k], pool & (player == who))
            measured = within(target, values[aspect], pool & (player == who))
            if not np.isnan(mine):
                verdicts.append(mine >= MIN_WITHIN and mine >= measured)
            add(aspect, f"1 within {who.upper()}", mine, MIN_WITHIN,
                None if np.isnan(mine) else mine >= MIN_WITHIN and mine >= measured,
                f"measure {measured:.3f}")  # fmt: skip
        add(aspect, "1 within players", float(all(verdicts)) if verdicts else np.nan, None,
            bool(verdicts) and all(verdicts), "both players" if len(verdicts) == 2 else
            f"{len(verdicts)} player(s) with {MIN_CLASS}+ serves per class")  # fmt: skip
        # 2. A player never trained on
        verdicts = []
        for who, (held, held_score) in left_out.items():
            area = within(target[held], held_score[:, k], np.ones(held.sum(), bool))
            if not np.isnan(area):
                verdicts.append(area >= MIN_NEW_PLAYER)
            add(aspect, f"2 new player {who.upper()}", area, MIN_NEW_PLAYER,
                None if np.isnan(area) else area >= MIN_NEW_PLAYER)  # fmt: skip
        add(aspect, "2 new players", float(all(verdicts)) if verdicts else np.nan, None,
            bool(verdicts) and all(verdicts))  # fmt: skip
        # 3. The professionals
        alarms = (score[pros, k] >= cuts[k]).mean()
        add(aspect, "3 professionals called needs work", alarms, MAX_PROS, alarms <= MAX_PROS,
            f"{pros.sum()} serves")  # fmt: skip
        # 4. Counterfactual edits on serves answered "as it should be"
        fine = (pool | test) & (target == 0)
        base = body.predict(m3, *body.relative(o[fine], m3["travel"]), prepared=True)[:, k]
        own = body.predict(
            m3, *body.relative(edited(aspect, o[fine]), m3["travel"]), prepared=True
        )[:, k]
        other = body.predict(m3, *body.relative(edited(UNRELATED[aspect], o[fine]), m3["travel"]),
                             prepared=True)[:, k]  # fmt: skip
        raised, moved = (own > base).mean(), np.abs(other - base).mean()
        add(aspect, "4 own edit raises", raised, MIN_RAISED, None,
            f"{fine.sum()} serves, mean change {np.mean(own - base):+.3f}")  # fmt: skip
        add(aspect, f"4 unrelated edit ({UNRELATED[aspect]}) moves", moved, MAX_MOVED, None)
        add(aspect, "4 counterfactual", float(raised >= MIN_RAISED and moved < MAX_MOVED), None,
            raised >= MIN_RAISED and moved < MAX_MOVED)  # fmt: skip
        # 5. Occlusion: share of the fall on the aspect's own part, against its share of joints
        positive = y[pool | test, k] == 1
        fall = np.abs(occluded[positive, :, k, :]).sum((0, 2))
        own_joints = [J(j) for j in PARTS[OWN[aspect]]]
        add(aspect, "5 occlusion on its own part", fall[own_joints].sum() / fall.sum(), None,
            None, f"{OWN[aspect]}: {len(own_joints)} of {len(fall)} joints")  # fmt: skip
        # 7. Calibration on validation
        have = pool & ~np.isnan(target)
        add(aspect, "7 Brier on validation", np.mean((score[have, k] - target[have]) ** 2))
        # 8. Fresh serves
        add(aspect, "8 test AUROC", within(target, score[:, k], test), None, None,
            f"{int((test & ~np.isnan(target)).sum())} answered")  # fmt: skip
        add(aspect, "8 fresh AUROC", within(target, score[:, k], fresh), None, None,
            f"{int((fresh & ~np.isnan(target)).sum())} answered, "
            f"{(score[fresh, k] >= cuts[k]).mean():.0%} called needs work")  # fmt: skip
    robustness(data, m3, names, cuts, test, add)
    plot_calibration(score[pool], y[pool], names, model_name)
    baseline.save(f"{model_name}-checks", rows)
    print(f"{model_name.upper()}, aspect by aspect (experiments/serve/{model_name}-checks.csv):")
    for r in rows:
        print(f"  {r['aspect']:15} {r['check']:38} {r['value']:7.3f}  line {r['line']!s:5} "
              f"{r['passes']:3} {r['note']}")  # fmt: skip
    print("Adopted (every line passes; the retest kappa is needed too):")
    for aspect in names:
        lines = [r for r in rows if r["aspect"] == aspect and r["check"] in (
            "0 retest kappa", "1 within players", "2 new players",
            "3 professionals called needs work", "4 counterfactual")]  # fmt: skip
        failed = [r["check"] for r in lines if r["passes"] == "no"]
        pending = [r["check"] for r in lines if r["passes"] == ""]
        verdict = "no" if failed else "pending" if pending else "yes"
        print(f"  {aspect}: {verdict}" + (f" (fails {', '.join(failed)})" if failed else ""))


def occlusion(m3: dict, x: np.ndarray, score) -> np.ndarray:
    """(windows, joints, aspects, stretches): fall of each probability when one joint moves like
    the reference serve (a serve with nothing to improve) during one stretch, as model.occlusion."""
    mask = np.ones(x.shape[:3], bool)
    base = body.predict(m3, x, mask, prepared=True)
    reference = m3["reference"].numpy()
    fall = np.zeros((len(x), x.shape[2], base.shape[1], len(model.STARTS)))
    for j in range(x.shape[2]):
        for s, start in enumerate(model.STARTS):
            hidden = model.replace(x, reference, j, start)
            fall[:, j, :, s] = base - body.predict(m3, hidden, mask, prepared=True)
    return fall


def plot_occlusion(fall: np.ndarray, y: np.ndarray, names: list[str], model_name: str) -> None:
    _, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    for k, (ax, aspect) in enumerate(zip(axes.flat, names)):
        image = fall[y[:, k] == 1, :, k, :].mean(0)
        limit = np.abs(image).max() or 1
        ax.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(f"{aspect} ({int((y[:, k] == 1).sum())} needing work)")
        ax.set_xticks(range(len(model.STARTS)), [f"{model.seconds(s):.1f}" for s in model.STARTS])
        ax.set_yticks(range(len(dataset.JOINTS)), dataset.JOINTS, fontsize=7)
        ax.set_xlabel("stretch start, s from contact")
    plt.suptitle(f"{model_name.upper()} occlusion: fall of 'needs work' when a joint moves like a clean serve "
                 "(left = tossing side)")  # fmt: skip
    plt.tight_layout()
    plt.savefig(baseline.RESULTS / f"{model_name}-occlusion.png", dpi=150)


def plot_calibration(score: np.ndarray, y: np.ndarray, names: list[str], model_name: str) -> None:
    _, ax = plt.subplots(figsize=(5, 5))
    edges = np.linspace(0, 1, 6)
    for k, aspect in enumerate(names):
        have = ~np.isnan(y[:, k])
        bins = np.clip(np.digitize(score[have, k], edges) - 1, 0, 4)
        said = [score[have, k][bins == b].mean() for b in range(5) if (bins == b).any()]
        seen = [y[have, k][bins == b].mean() for b in range(5) if (bins == b).any()]
        ax.plot(said, seen, "o-", label=aspect)
    ax.plot([0, 1], [0, 1], "k:", label="calibrated")
    ax.set(xlabel="probability on validation", ylabel="share answered needs work",
           title=f"{model_name.upper()} calibration")  # fmt: skip
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(baseline.RESULTS / f"{model_name}-calibration.png", dpi=150)


def robustness(data, m3: dict, names: list[str], cuts: np.ndarray, test, add) -> None:
    """6. A mirrored window, points unseen and the camera in front of the player."""
    x, mask = data["x"][test], data["mask"][test]
    base = body.predict(m3, x, mask)
    mirrored, mirrored_mask = x.copy(), mask.copy()
    train.mirror(mirrored, mirrored_mask, np.ones(len(x), bool))
    change = np.abs(body.predict(m3, mirrored, mirrored_mask) - base).max(0)
    rng = np.random.default_rng(0)
    unseen = {share: body.predict(m3, *bias.unsee(x, mask, share, rng, np.arange(x.shape[2])))
              for share in (0.1, 0.2, 0.3, 0.4)}  # fmt: skip
    thetis = np.load(pretrain.THETIS) if pretrain.THETIS.exists() else None
    if thetis is not None:
        said = body.predict(m3, thetis["x"], thetis["mask"]) >= cuts
        side = model.shoulder_width(thetis["x"]) >= model.MIN_SHOULDERS
    for k, aspect in enumerate(names):
        add(aspect, "6 mirrored, largest change", change[k])
        worst = max(unseen, key=lambda s: np.abs(unseen[s][:, k] - base[:, k]).mean())
        add(aspect, "6 points unseen, largest mean change",
            np.abs(unseen[worst][:, k] - base[:, k]).mean(), None, None, f"at {worst:.0%} unseen")  # fmt: skip
        if thetis is not None:
            add(aspect, "6 THETIS (front) called needs work", said[:, k].mean(), None, None,
                f"{said[side, k].mean():.0%} of the {side.sum()} the app keeps")  # fmt: skip


if __name__ == "__main__":
    main()
