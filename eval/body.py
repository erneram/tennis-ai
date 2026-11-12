"""How far the team's answers about the body can be trusted, with no second annotator.

    uv run python -m eval.body [--name ernesto] [--checks [--model m3b] | --ablation | --levels | --recommend]

Reads data/processed/serve_dataset.npz (python -m strokes.serve.dataset) and the answers of the
"Cuerpo" tab, eval/serve/aspects_<name>.csv. For each aspect of strokes/serve/body.ANSWERS and
each version of its measure (as first written, and as the audit of Nov 12 left it):

1. Answer against measure: AUROC of the pose measure (strokes/serve/body.MEASURES, higher means
   more to improve, with its direction fixed beforehand) for the answer "needs work", on every
   answered serve and within EA and within JC alone, where each class has MIN_CLASS serves or
   more. Within a player it says whether the answer follows the serve and not who serves.
2. The professionals: the annotator called every professional serve "as it should be". The
   measure's AUROC for amateurs against professionals, the 90th percentile of the professionals
   trained on (their range), and past it the share of the held-out professionals and of the
   amateur serves, with how many of those the annotator marked.
3. The annotator against themself: once "Cuerpo · repetir" has answers
   (aspects_<name>_retest.csv), Cohen's kappa between both passes on the needs-work answer, and
   on the raw answer. An aspect with a kappa under MIN_KAPPA is not used to train or shown.

Writes experiments/serve/body.csv.

--checks scores a body model (python -m strokes.serve.body: MODELS), each aspect on its own, and
writes experiments/serve/<model>-checks.csv, -left-out.csv, -occlusion.png and -calibration.png:

1. Within each player: validation AUROC within EA and within JC, against the measure's on the
   same serves (the higher of its two versions).
2. A new player: the model retrained without EA, scored on EA; and without JC, scored on JC.
3. The professionals (pro01 to pro05, Federer included): share called "needs work".
4. Counterfactual: on serves answered "as it should be", the part of the body the aspect asks
   about is moved toward "needs work" (edited); its probability should rise, and should barely
   move when an unrelated part is moved instead (UNRELATED).
5. Occlusion: the share of the probability that rests on the aspect's own part (OWN, PARTS).
6. Robustness: a mirrored window, 10 to 40 % of the points unseen, the camera in front (THETIS).
7. Calibration: Brier score and reliability curve on validation.
8. Fresh serves: the test split and the YouTube clips of Nov 9 (body.FRESH).
9. Precision: of the amateur serves it calls "needs work" on validation, the share marked.

An aspect is validated only if it passes every line fixed before M3 was trained (SUMMARY):
retest kappa MIN_KAPPA or more; within EA and within JC (where both classes have MIN_CLASS
serves) AUROC MIN_WITHIN or more and no lower than its measure's; on the player left out, AUROC
MIN_NEW_PLAYER or more; at most MAX_PROS of the professionals called "needs work"; and its own
edit raising the probability on MIN_RAISED of the serves while the unrelated edit moves it less
than MAX_MOVED on average. Checks 5 to 8 are reported, not lines. --levels adds the two levels
under it the app may also show, written after M3 and M3b's checks (level, levels), and writes
experiments/serve/points.csv, which the app reads. --ablation is a diagnosis (ablation).

--recommend scores the app's list of points (the toss arm by M1e and the aspects of points.csv; at
most MAX_POINTS, the surest level first and within it the furthest past its threshold; none means
"no clear problem") against the annotator's answers, next to M1e alone (the app before), the same
list with EA and JC scored by models trained without them, every aspect of each model, the pose
measures alone and toss_hold always. Per list, by split and player: precision (the points the
annotator marked), coverage (the amateurs' marked problems that got a point), professional serves
told "no clear problem", amateur serves with two or more marked problems told the same, how often
the first point is the marked problem the measures find clearest, and the value per amateur
analysis in quetzales (configs/value.yaml: one hit per right point, one false alarm per wrong one,
one miss per marked problem left out). Writes experiments/serve/recommend.csv.
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from core import ROOT
from eval import bias, improvement, value
from eval.run_eval import kappa
from strokes.serve import baseline, body, dataset, model, pretrain, train

EVAL = ROOT / "eval" / "serve"
MIN_CLASS = 5
MIN_KAPPA = body.MIN_KAPPA
MIN_WITHIN, MIN_NEW_PLAYER, MAX_PROS, MIN_RAISED, MAX_MOVED = 0.70, 0.65, 0.20, 0.80, 0.05
MIN_PRECISION = 0.70  # of the amateur serves a model or a measure flags, the share marked
MODELS = ("m3", "m3b", "m3c", "m4")  # the body models --levels chooses among, by their checks
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
SUMMARY = ("0 retest kappa", "1 within players", "2 new players",
           "3 professionals called needs work", "4 counterfactual")  # fmt: skip
# The lines of "on trial" (level): SUMMARY without "1 within players", which is asked of one
# player only and not against the measure, plus the precision
TRIAL = ("0 retest kappa", "2 new players", "3 professionals called needs work",
         "4 counterfactual", "9 precision among amateurs")  # fmt: skip
MAX_POINTS = 3  # points to improve one analysis gives at most
RANK = {"validated": 2, "trial": 1, "measure": 0}  # the order of the points: the surest first
MIN_SEPARATES = body.MIN_SEPARATES
MEASURED = ("auroc", "auroc_ea", "auroc_jc", "median_needs_work", "median_fine", "median_pros",
            "pros_like_needs_work", "auroc_pros", "pros_p90", "outside_amateurs",
            "outside_precision", "held_pros_outside", "pool_sd")  # fmt: skip


def within(positive: np.ndarray, score: np.ndarray, chosen: np.ndarray) -> float:
    """AUROC among the chosen serves, NaN when a class has fewer than MIN_CLASS."""
    have = chosen & ~np.isnan(positive) & ~np.isnan(score)
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
    parser.add_argument("--checks", action="store_true", help="score a body model aspect by aspect")
    parser.add_argument("--levels", action="store_true", help="what the app shows, and how sure")
    parser.add_argument("--recommend", action="store_true", help="score the lists of points")
    parser.add_argument("--ablation", action="store_true", help="M3c's two changes, one by one")
    parser.add_argument("--model", default="m3", help="one of MODELS")
    args = parser.parse_args()
    data = np.load(dataset.OUT)
    y = body.targets(body.answers(EVAL / f"aspects_{args.name}.csv"), data["video"])
    kappas = retest(args.name)
    rows = measured(data, y, kappas)
    print(f"{args.name}'s answers against the pose measures, {(~np.isnan(y)).any(1).sum()} serves")
    for row in rows:
        print(f"  {row['aspect']} ({row['version']}): {row['needs_work']} of {row['answered']} need "
              f"work; AUROC {row['auroc']:.2f}, within EA {row['auroc_ea']:.2f}, within JC "
              f"{row['auroc_jc']:.2f}; amateurs against pros {row['auroc_pros']:.2f}; past the "
              f"pros' 90th percentile: {row['outside_amateurs']} amateur serves, "
              f"{row['outside_precision']:.0%} of them marked, and {row['held_pros_outside']:.0%} "
              f"of the pros held out")  # fmt: skip
        if row["version"] == "first" and row["aspect"] in kappas:
            print(f"    retest on {row['retest_serves']} serves: kappa {row['retest_kappa']:.2f} "
                  f"(raw answer {row['retest_kappa_raw']:.2f})"
                  + (" under the line, not used" if row["retest_kappa"] < MIN_KAPPA else ""))  # fmt: skip
    baseline.save("body", rows)
    if args.checks:
        checks(data, y, kappas, args.model)
    if args.ablation:
        ablation(data, y)
    if args.levels:
        levels()
    if args.recommend:
        recommend(data, y)


def measured(data, y: np.ndarray, kappas: dict) -> list[dict]:
    """A row per aspect and version of its measure (body.measures as first written, then as the
    audit left it): how it follows the answers, and how it reads against the professionals."""
    judged, player = data["valid"] == 1, data["player"]
    pros = judged & np.char.startswith(player, "pro")
    amateurs = judged & ~pros
    chosen = body.splits(data, y)
    pool, held = chosen["pool"], chosen["test"] | chosen["fresh"]
    rows = []
    for version, audited in (("first", False), ("audited", True)):
        values = body.measures(data["x"], data["mask"], audited)
        for k, (aspect, score) in enumerate(values.items()):
            target = y[:, k]
            answered = judged & ~np.isnan(target)
            needs = answered & (target == 1)
            row = {"aspect": aspect, "version": version,
                   "measure": (body.AUDITED if audited else {}).get(aspect, body.MEASURES[aspect]),
                   "answered": int(answered.sum()), "needs_work": int(needs.sum())}  # fmt: skip
            if np.isnan(score).all():  # the audit left no measure for it
                rows.append(row | dict.fromkeys(MEASURED, np.nan))
                continue
            top = np.percentile(score[pool & pros], 90)  # higher means more to improve
            outside = pool & amateurs & answered & (score > top)
            rows.append(row | {
                "auroc": within(target, score, answered),
                "auroc_ea": within(target, score, judged & (player == "ea")),
                "auroc_jc": within(target, score, judged & (player == "jc")),
                "median_needs_work": float(np.median(score[needs & amateurs])),
                "median_fine": float(np.median(score[answered & (target == 0) & amateurs])),
                "median_pros": float(np.median(score[pros])),
                "pros_like_needs_work": float(
                    (score[pros] >= np.median(score[needs & amateurs])).mean()
                ),
                "auroc_pros": baseline.auroc(amateurs[judged], score[judged]),
                "pros_p90": float(top),
                "outside_amateurs": int(outside.sum()),
                "outside_precision": float((target[outside] == 1).mean()) if outside.any() else np.nan,
                "held_pros_outside": float((score[held & pros] > top).mean()),
                "pool_sd": float(np.std(score[pool])),
            })  # fmt: skip
    for row in rows:
        kappa_, raw, serves = kappas.get(row["aspect"], (np.nan, np.nan, 0))
        row |= {"retest_kappa": kappa_, "retest_kappa_raw": raw, "retest_serves": serves}
    return rows


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


def new_player(data, o, y, chosen: dict, player: str, m3: dict):
    """The serves of `player` and the model's scores for them, trained again without that player
    (on body.oriented() windows `o`)."""
    fit_ = chosen["pool"] & (data["player"] != player)
    held = (chosen["pool"] | chosen["test"]) & (data["player"] == player)
    scores = []
    for seed in range(len(m3["networks"])):
        networks, _ = body.fit_heads(o[fit_], y[fit_], seed, m3["epochs"], m3,
                                     data["player"][fit_])  # fmt: skip
        scores.append(body.probabilities(networks, o[held], m3))
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
    # The yardstick of line 1: the measure as first written and as the audit left it
    yardsticks = [body.measures(data["x"], data["mask"], audited) for audited in (False, True)]
    o = body.oriented(data["x"], data["mask"], m3["at_reach"])[0]
    cuts = np.array([m3["thresholds"][a] for a in names])
    rows = []

    def add(aspect, check, value, line=None, passes=None, note=""):
        rows.append({"aspect": aspect, "check": check, "value": round(float(value), 3),
                     "line": "" if line is None else line,
                     "passes": "" if passes is None else "yes" if passes else "no", "note": note})  # fmt: skip

    left_out = {p: new_player(data, o, y, chosen, p, m3) for p in ("ea", "jc")}
    baseline.save(f"{model_name}-left-out", [
        {"model": model_name, "video": v, "player": who, "aspect": aspect,
         "probability": round(float(p), 5)}
        for who, (held, held_score) in left_out.items()
        for v, values in zip(data["video"][held], held_score)
        for aspect, p in zip(names, values)])  # fmt: skip
    judged = pool | test | fresh
    pros = judged & np.char.startswith(player, "pro")
    if not m3["parts"]:
        occluded = occlusion(m3, body.prepared(o[pool | test], m3["travel"])["body"][0])
        plot_occlusion(occluded, y[pool | test], names, model_name)
    for k, aspect in enumerate(names):
        target = y[:, k]
        # 0. The annotator against themself
        found = aspect in kappas and not np.isnan(kappas[aspect][0])
        add(aspect, "0 retest kappa", kappas[aspect][0] if found else np.nan, MIN_KAPPA,
            kappas[aspect][0] >= MIN_KAPPA if found else None,
            "" if found else "pending: Cuerpo · repetir")  # fmt: skip
        # 1. Within each player, against the measure
        passed = []
        for who in ("ea", "jc"):
            mine = within(target, score[:, k], pool & (player == who))
            first, audited = (within(target, v[aspect], pool & (player == who)) for v in yardsticks)
            measured = np.nanmax([first, audited]) if not np.isnan(mine) else np.nan
            if not np.isnan(mine):
                passed.append(mine >= MIN_WITHIN and mine >= measured)
            add(aspect, f"1 within {who.upper()}", mine, MIN_WITHIN,
                None if np.isnan(mine) else mine >= MIN_WITHIN and mine >= measured,
                f"measure {first:.3f}, audited {audited:.3f}")  # fmt: skip
        add(aspect, "1 within players", float(all(passed)) if passed else np.nan, None,
            bool(passed) and all(passed), "both players" if len(passed) == 2 else
            f"{len(passed)} player(s) with {MIN_CLASS}+ serves per class")  # fmt: skip
        # 2. A player never trained on
        passed = []
        for who, (held, held_score) in left_out.items():
            area = within(target[held], held_score[:, k], np.ones(held.sum(), bool))
            if not np.isnan(area):
                passed.append(area >= MIN_NEW_PLAYER)
            add(aspect, f"2 new player {who.upper()}", area, MIN_NEW_PLAYER,
                None if np.isnan(area) else area >= MIN_NEW_PLAYER)  # fmt: skip
        add(aspect, "2 new players", float(all(passed)) if passed else np.nan, None,
            bool(passed) and all(passed))  # fmt: skip
        # 3. The professionals
        alarms = (score[pros, k] >= cuts[k]).mean()
        add(aspect, "3 professionals called needs work", alarms, MAX_PROS, alarms <= MAX_PROS,
            f"{pros.sum()} serves")  # fmt: skip
        # 4. Counterfactual edits on serves answered "as it should be"
        fine = (pool | test) & (target == 0)
        base = body.predict_oriented(m3, o[fine])[:, k]
        own = body.predict_oriented(m3, edited(aspect, o[fine]))[:, k]
        other = body.predict_oriented(m3, edited(UNRELATED[aspect], o[fine]))[:, k]
        raised, moved = (own > base).mean(), np.abs(other - base).mean()
        add(aspect, "4 own edit raises", raised, MIN_RAISED, None,
            f"{fine.sum()} serves, mean change {np.mean(own - base):+.3f}")  # fmt: skip
        add(aspect, f"4 unrelated edit ({UNRELATED[aspect]}) moves", moved, MAX_MOVED, None)
        add(aspect, "4 counterfactual", float(raised >= MIN_RAISED and moved < MAX_MOVED), None,
            raised >= MIN_RAISED and moved < MAX_MOVED)  # fmt: skip
        # 5. Occlusion: share of the fall on the aspect's own part, against its share of joints
        if m3["parts"]:
            add(aspect, "5 occlusion on its own part", 1, None, None,
                "by construction: its network sees only its part")  # fmt: skip
        else:
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
        # 9. Of the amateur serves it calls "needs work" on validation, the share marked
        said = pool & ~pros & ~np.isnan(target) & (score[:, k] >= cuts[k])
        precision = (target[said] == 1).mean() if said.any() else np.nan
        add(aspect, "9 precision among amateurs", precision, MIN_PRECISION,
            bool(precision >= MIN_PRECISION), f"{said.sum()} serves called needs work")  # fmt: skip
    robustness(data, m3, names, cuts, test, add)
    plot_calibration(score[pool], y[pool], names, model_name)
    baseline.save(f"{model_name}-checks", rows)
    print(f"{model_name.upper()}, aspect by aspect (experiments/serve/{model_name}-checks.csv):")
    for r in rows:
        print(f"  {r['aspect']:15} {r['check']:38} {r['value']:7.3f}  line {r['line']!s:5} "
              f"{r['passes']:3} {r['note']}")  # fmt: skip
    print("Validated (every line of SUMMARY) or on trial (level):")
    for aspect, (verdict, failed) in verdicts(rows).items():
        print(f"  {aspect}: {level(rows, aspect) or 'neither'}"
              + (f" (validated: {verdict}, fails {', '.join(failed)})" if failed else ""))  # fmt: skip


def verdicts(rows: list[dict]) -> dict[str, tuple[str, list[str]]]:
    """Per aspect of a checks table: "yes", "no" or "pending" by the SUMMARY lines, and the lines
    it fails."""
    out = {}
    for aspect in dict.fromkeys(r["aspect"] for r in rows):
        lines = [r for r in rows if r["aspect"] == aspect and r["check"] in SUMMARY]
        failed = [r["check"] for r in lines if r["passes"] == "no"]
        pending = any(r["passes"] == "" for r in lines)
        out[aspect] = ("no" if failed else "pending" if pending else "yes", failed)
    return out


def ablation(data, y_all: np.ndarray) -> None:
    """M3c's two changes to M3b one at a time, to experiments/serve/m3c-ablation.csv: contact at
    the reach, and the loss balanced by player. Per aspect, the AUROC of lines 1 and 2. A
    diagnosis of why M3c does not carry over as M3b does, not models the app may use."""
    names = body.kept()
    y = y_all[:, [list(body.ANSWERS).index(a) for a in names]]
    chosen, player = body.splits(data, y), data["player"]
    pool, rows = chosen["pool"], []
    for at_reach, balanced in ((False, False), (True, False), (False, True), (True, True)):
        spec = {"aspects": names, "travel": True, "at_reach": at_reach, "balanced": balanced,
                "parts": False, "epochs": 85, "networks": [None] * 5}  # fmt: skip
        o = body.oriented(data["x"], data["mask"], at_reach)[0]
        validation = np.zeros((len(spec["networks"]), *y[pool].shape))
        for seed in range(len(validation)):
            fold = body.session_folds(data["video"][pool], seed)
            for k in range(train.FOLDS):
                networks, _ = body.fit_heads(o[pool][fold != k], y[pool][fold != k], seed,
                                             spec["epochs"], spec, player[pool][fold != k])  # fmt: skip
                validation[seed, fold == k] = body.probabilities(networks, o[pool][fold == k], spec)
        score = np.full(y.shape, np.nan)
        score[pool] = validation.mean(0)
        left_out = {p: new_player(data, o, y, chosen, p, spec) for p in ("ea", "jc")}
        for k, aspect in enumerate(names):
            row = {"contact": "reach" if at_reach else "found",
                   "loss": "by player" if balanced else "by class", "aspect": aspect}  # fmt: skip
            for who, (held, held_score) in left_out.items():
                row[f"within_{who}"] = within(y[:, k], score[:, k], pool & (player == who))
                row[f"new_{who}"] = within(y[held, k], held_score[:, k], np.ones(held.sum(), bool))
            rows.append(row)
            print(f"  contact {row['contact']:5}, loss {row['loss']:9}, {aspect:15} within EA "
                  f"{row['within_ea']:.3f}, within JC {row['within_jc']:.3f}, new EA "
                  f"{row['new_ea']:.3f}, new JC {row['new_jc']:.3f}", flush=True)  # fmt: skip
    baseline.save("m3c-ablation", rows)


def level(rows: list[dict], aspect: str) -> str:
    """How far a model's answer for one aspect can be trusted, from its checks table: "validated"
    when it passes every line fixed before M3 was trained (SUMMARY); "trial" when it passes those
    of TRIAL and ranks the serves of at least one player (AUROC MIN_WITHIN or more within EA or
    within JC, not against the measure); else ""."""
    mine = {r["check"]: r for r in rows if r["aspect"] == aspect}
    if all(mine[check]["passes"] == "yes" for check in SUMMARY):
        return "validated"
    within_one = any(float(mine[f"1 within {who}"]["value"]) >= MIN_WITHIN for who in ("EA", "JC"))
    on_trial = all(mine.get(check, {}).get("passes") == "yes" for check in TRIAL)
    return "trial" if on_trial and within_one else ""


def levels() -> None:
    """What the app may say about each aspect and how sure it is, to experiments/serve/points.csv.
    A model's answer, validated or on trial (level), from the model of MODELS with the highest
    level for the aspect (ties: the one that ranks a player it never saw best, its lowest AUROC of
    check 2). Else its audited measure against the professionals' range, when all hold: retest
    kappa MIN_KAPPA or more; amateurs against professionals AUROC MIN_SEPARATES or more; past the
    90th percentile of the professionals trained on, at most MAX_PROS of those held out (pro01,
    pro05) and MIN_PRECISION or more of the amateur serves there marked "needs work"; and the
    measure not contradicting the answers within a player (AUROC 0.5 or more where there is one).
    Written on Nov 12, after M3 and M3b's checks and before M3c and M4 were trained. `described`
    marks the measures the app shows next to the professionals' without a point, by the rule from
    before M3: retest kappa MIN_KAPPA or more and amateurs against professionals MIN_SEPARATES."""
    with open(baseline.RESULTS / "body.csv", newline="") as file:
        measure = {r["aspect"]: r for r in csv.DictReader(file) if r["version"] == "audited"}
    tables = {}
    for name in MODELS:
        path = baseline.RESULTS / f"{name}-checks.csv"
        if path.exists():
            with open(path, newline="") as file:
                tables[name] = list(csv.DictReader(file))
    number = lambda text: float(text) if text not in ("", "nan") else np.nan
    order = {"validated": 2, "trial": 1, "": 0}
    rows = []
    for aspect in body.ANSWERS:
        best = ("", "", -np.inf)  # level, model, lowest AUROC on a player left out
        for name, table in tables.items():
            mine = {r["check"]: r for r in table if r["aspect"] == aspect}
            if not mine:
                continue
            found = level(table, aspect)
            new = np.nanmin([number(mine[f"2 new player {who}"]["value"]) for who in ("EA", "JC")]
                            + [np.inf])  # fmt: skip
            if (order[found], new) > (order[best[0]], best[2]):
                best = (found, name, new)
        m = {k: number(v) for k, v in measure[aspect].items() if k in MEASURED + ("retest_kappa",)}
        fails = ["no measure"] if np.isnan(m["auroc_pros"]) else [text for text, bad in (
            ("retest kappa", not m["retest_kappa"] >= MIN_KAPPA),
            ("amateurs against pros", not m["auroc_pros"] >= MIN_SEPARATES),
            ("pros held out", not m["held_pros_outside"] <= MAX_PROS),
            ("precision", not m["outside_precision"] >= MIN_PRECISION),
            ("contradicts the answers within a player",
             np.nanmin([m["auroc_ea"], m["auroc_jc"], np.inf]) < 0.5),
        ) if bad]  # fmt: skip
        described = m["retest_kappa"] >= MIN_KAPPA and m["auroc_pros"] >= MIN_SEPARATES
        row = {"aspect": aspect, "level": "", "source": "", "model": "", "threshold": "",
               "precision": "", "pros_rate": "", "new_player": "",
               "described": "yes" if described else "", "pros_median": m["median_pros"],
               "pros_p90": m["pros_p90"], "scale": m["pool_sd"],
               "measure_fails": "; ".join(fails)}  # fmt: skip
        if best[0]:
            mine = {r["check"]: r for r in tables[best[1]] if r["aspect"] == aspect}
            row |= {"level": best[0], "source": "model", "model": best[1],
                    "precision": mine["9 precision among amateurs"]["value"],
                    "pros_rate": mine["3 professionals called needs work"]["value"],
                    "new_player": round(best[2], 3)}  # fmt: skip
        elif not fails:
            row |= {"level": "measure", "source": "measure", "threshold": m["pros_p90"],
                    "precision": round(m["outside_precision"], 3),
                    "pros_rate": round(m["held_pros_outside"], 3)}  # fmt: skip
        rows.append(row)
    baseline.save("points", rows)
    print("What the app says about each aspect (experiments/serve/points.csv):")
    for r in rows:
        print(f"  {r['aspect']:15} {r['level'] or 'nothing':9} {r['model'] or r['source']:8}"
              + (f" precision {float(r['precision']):.2f}, pros {float(r['pros_rate']):.0%}"
                 if r["level"] else f" measure fails: {r['measure_fails']}"))  # fmt: skip


def pick(margin: np.ndarray, allowed: np.ndarray, rank=None) -> tuple[np.ndarray, np.ndarray]:
    """Which problems each serve gets a point for: at most MAX_POINTS of the allowed ones with a
    margin past 0, the highest `rank` first (RANK, one per column) and within it the furthest
    past; and the column of the first."""
    margin = np.where(allowed & ~np.isnan(margin), margin, -np.inf)
    key = np.where(margin >= 0, margin + 1e3 * (0 if rank is None else np.asarray(rank)), -np.inf)
    order = np.argsort(-key, 1, kind="stable")[:, :MAX_POINTS]
    said = np.zeros(margin.shape, bool)
    said[np.arange(len(margin))[:, None], order] = np.take_along_axis(key, order, 1) > -np.inf
    return said, order[:, 0]


def shown() -> dict[str, dict]:
    """The aspects of experiments/serve/points.csv (levels) the app gives a point for."""
    with open(baseline.RESULTS / "points.csv", newline="") as file:
        return {r["aspect"]: r for r in csv.DictReader(file) if r["level"]}


def recommend(data, y_all: np.ndarray) -> None:
    """Scores the lists of points against the annotator's answers and the toss labels, to
    experiments/serve/recommend.csv: by split (validation, test, fresh) and by player."""
    names = body.kept()
    y = y_all[:, [list(body.ANSWERS).index(a) for a in names]]
    toss = np.where(data["valid"] == 1, data["category"] == baseline.POSITIVE, np.nan)
    truth = np.column_stack([toss, y])  # the toss arm, then each aspect: 1 marked, NaN unanswered
    chosen, player = body.splits(data, y), data["player"]
    pool, amateurs = chosen["pool"], ~np.char.startswith(player, "pro")
    toss_margin = improvement.toss_probability(data) - bias.thresholds()["m1e"]
    audited = body.measures(data["x"], data["mask"], audited=True)
    measured = np.column_stack([-body.toss_height(data["x"], data["mask"]),
                                *[audited[a] for a in names]])  # fmt: skip
    cuts = [0.0] + [np.nan if np.isnan(audited[a]).all() else
                    body.threshold(y[pool, k], audited[a][pool]) for k, a in enumerate(names)]  # fmt: skip
    scale = np.nanstd(measured[pool], 0)  # comparable across aspects
    clearest = np.where(truth == 1, np.nan_to_num((measured - cuts) / scale, nan=-np.inf),
                        -np.inf).argmax(1)  # fmt: skip
    every = np.ones(truth.shape[1], bool)

    def margins(name: str, left_out: bool = False) -> np.ndarray:
        """Probability of each aspect of `names` past its threshold by one model, on serves it
        was not trained on; with `left_out`, EA's and JC's by the model trained without them."""
        m3, scored = body.load(name), oos(name, body.load(name)["aspects"])
        if left_out:
            with open(baseline.RESULTS / f"{name}-left-out.csv", newline="") as file:
                rows = list(csv.DictReader(file))
            for video in {r["video"] for r in rows}:
                mine = {r["aspect"]: float(r["probability"]) for r in rows if r["video"] == video}
                scored[video] = np.array([mine[a] for a in m3["aspects"]])
        score = np.array(
            [scored.get(v, np.full(len(m3["aspects"]), np.nan)) for v in data["video"]]
        )
        return np.column_stack([score[:, m3["aspects"].index(a)] - m3["thresholds"][a]
                                if a in m3["aspects"] else np.full(len(score), np.nan)
                                for a in names])  # fmt: skip

    def app(left_out: bool = False):
        """The app's list: the toss arm by M1e and each aspect of points.csv by its model or
        its measure against the professionals' range, the surest level first."""
        points, margin, rank = shown(), [toss_margin], [RANK["validated"]]
        for k, aspect in enumerate(names):
            point = points.get(aspect)
            if not point:
                margin.append(np.full(len(truth), np.nan))
            elif point["source"] == "model":
                margin.append(margins(point["model"], left_out)[:, k])
            else:
                margin.append((audited[aspect] - float(point["threshold"])) / float(point["scale"]))
            rank.append(RANK[point["level"]] if point else 0)
        return np.column_stack(margin), every, rank

    models = [m for m in MODELS if (baseline.RESULTS / f"{m}-left-out.csv").exists()]
    lists = {
        "M1e alone (the app before)": (toss_margin[:, None] * np.eye(1, truth.shape[1]), every, None),
        "the app: M1e and the aspects of points.csv": app(),
        "the app, EA and JC by models that never saw them": app(left_out=True),
        **{f"M1e and every {m.upper()} aspect": (np.column_stack([toss_margin, margins(m)]), every,
                                                 None) for m in models},
        "pose measures only": ((measured - cuts) / scale, every, None),
        "toss_hold always": (np.eye(1, truth.shape[1]) * np.ones((len(truth), 1)), every, None),
    }  # fmt: skip
    lists["M1e alone (the app before)"][0][:, 1:] = np.nan
    lists["toss_hold always"][0][:, 1:] = np.nan
    config = yaml.safe_load(value.CONFIG.read_text())
    block = config["coach_per_hour"] * config["drill_minutes"] / 60
    groups = {"all": np.ones(len(truth), bool), "jc": player == "jc", "ea": player == "ea",
              "pros": ~amateurs}  # fmt: skip
    rows = []
    for title, (margin, allowed, rank) in lists.items():
        said, first = pick(margin, allowed, rank)
        right, wrong, missed = said & (truth == 1), said & (truth == 0), ~said & (truth == 1)
        several = (truth == 1).sum(1) >= 2  # the annotator marked two problems or more
        for split, group in [(s, g) for s in ("pool", "test", "fresh") for g in groups]:
            mine = chosen[split] & groups[group]
            ama = mine & amateurs
            if not mine.any():
                continue
            told = mine & said.any(1) & (truth == 1).any(1)
            gain = (config["hit"] * right - config["false_alarm"] * wrong
                    - config["miss"] * missed)[ama].sum(1)  # fmt: skip
            rows.append({
                "list": title, "split": "validation" if split == "pool" else split, "player": group,
                "serves": int(mine.sum()), "points": int(said[mine].sum()),
                "precision": right[mine].sum() / max((right | wrong)[mine].sum(), 1),
                "coverage": right[ama].sum() / max((truth[ama] == 1).sum(), 1),
                "pros_clear": float((~said[mine & ~amateurs].any(1)).mean()) if (mine & ~amateurs).any() else np.nan,
                "several_marked": int((ama & several).sum()),
                "several_told_clear": int((ama & several & ~said.any(1)).sum()),
                "first_is_clearest": float((first[told] == clearest[told]).mean()) if told.any() else np.nan,
                "value": (block * gain.mean() - value.compute_cost(config, config["pose_seconds"])
                          if ama.any() else np.nan),
            })  # fmt: skip
    baseline.save("recommend", rows)
    print(f"Lists of points, at most {MAX_POINTS} per serve (experiments/serve/recommend.csv); "
          f"the app shows: {', '.join(f'{a} ({r['level']})' for a, r in shown().items()) or 'the toss arm only'}")  # fmt: skip
    for r in rows:
        if r["player"] == "all" or r["list"].startswith("the app"):
            print(f"  {r['list']:48} {r['split']:10} {r['player']:4} {r['serves']:3} serves, "
                  f"{r['points']:3} points: precision {r['precision']:.2f}, coverage "
                  f"{r['coverage']:.2f}, pros told no problem {r['pros_clear']:.2f}, told no "
                  f"problem with 2+ marked {r['several_told_clear']} of {r['several_marked']}, "
                  f"Q{r['value']:.2f} per amateur analysis")  # fmt: skip


def occlusion(m3: dict, x: np.ndarray) -> np.ndarray:
    """(windows, joints, aspects, stretches): fall of each probability when one joint moves like
    the reference serve (a serve with nothing to improve) during one stretch, as model.occlusion.
    For a model with one network on the whole body, on its input `x`."""
    mask = np.ones(x.shape[:3], bool)
    predict = lambda x: np.mean([body.probability(n["body"], x, mask) for n in m3["networks"]], 0)
    base = predict(x)
    reference = m3["reference"].numpy()
    fall = np.zeros((len(x), x.shape[2], base.shape[1], len(model.STARTS)))
    for j in range(x.shape[2]):
        for s, start in enumerate(model.STARTS):
            fall[:, j, :, s] = base - predict(model.replace(x, reference, j, start))
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
