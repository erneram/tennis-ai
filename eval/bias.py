"""Bias and robustness of the serve models (written report, section 4.8).

    uv run python -m eval.bias

Every model at the threshold eval/value.py chose for it on validation scores:

1. Errors by group, on the validation (out-of-fold) and test scores together: amateur (JC) or
   professional, frame rate and orientation of the recording, and how much of the arms MediaPipe
   saw. The groups are small: the counts are printed with every rate.
2. Within JC: AUROC on JC's serves only, against all serves. JC is the only amateur and every serve
   with the fault is his, so a model can score well by telling JC from the professionals.
3. Unseen points: a share of the arm points of each test window is set unseen and its position
   interpolated in time, as dataset.window does with points MediaPipe missed (worse light, a
   farther camera); how the probability and the share of "drops early" move.
4. Mirror: left- and right-handed players. The arm models put the tossing arm on the same side, so
   a mirrored window must give the same probability.
5. Camera from the front: the share of THETIS serves (front view, never labeled, data/processed/
   thetis_dataset.npz; skipped when absent) each model calls "drops early", beginners (players
   p1 to p31) against experts (p32 to p55), and the share the app's camera check (shoulder width
   in the stance) sends back to be recorded again.

Writes experiments/serve/bias.csv (rates by group) and bias.png (parts 3 and 5).
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from strokes.serve import baseline, dataset, model, pretrain, train

MODELS = ["b0", "m1c", "m2c"]  # the rule, the best model from scratch, the app's model
SHARES = [0.0, 0.1, 0.2, 0.3, 0.4]  # of the arm points set unseen
REPEATS = 20
ARM_JOINTS = np.array(train.ARMS)


def scores(name: str) -> dict[tuple[str, str], np.ndarray]:
    """(split, video) -> scores over seeds, from experiments/serve/<name>.csv."""
    out: dict[tuple[str, str], list[float]] = {}
    for r in csv.DictReader((baseline.RESULTS / f"{name}.csv").open()):
        if r["split"] in ("validation", "test"):
            out.setdefault((r["split"], r["video"]), []).append(float(r["score"]))
    return {k: np.array(v) for k, v in out.items()}


def thresholds() -> dict[str, float]:
    return {r["model"]: float(r["threshold"])
            for r in csv.DictReader((baseline.RESULTS / "value.csv").open())}  # fmt: skip


def groups(data, i: int, arms_seen: float) -> dict[str, str]:
    """The groups of one video of the dataset (its orientation comes from eval/eda.csv)."""
    return {
        "player": "professional" if data["player"][i].startswith("pro") else "amateur (JC)",
        "frame rate": "60 fps" if data["video_fps"][i] > 45 else "30 fps",
        "arm points seen": "90 % or more" if arms_seen >= 0.9 else "under 90 %",
    }


def by_group(data, names: list[str], cuts: dict) -> list[dict]:
    """Recall and false-alarm rate of each model in each group (mean over seeds)."""
    arms_seen = train.arms(data["x"], data["mask"])[1].mean((1, 2))
    orientation = {r["video"]: "portrait" if int(r["height"]) > int(r["width"]) else "landscape"
                   for r in csv.DictReader((baseline.RESULTS / "eda.csv").open())}  # fmt: skip
    index = {v: i for i, v in enumerate(data["video"])}
    rows = []
    for name in names:
        for (split, video), score in scores(name).items():
            i = index[video]
            said = (score >= cuts[name]).mean()
            positive = data["category"][i] == baseline.POSITIVE
            for group, value in {**groups(data, i, arms_seen[i]),
                                 "orientation": orientation[video]}.items():  # fmt: skip
                rows.append({"model": name, "group": group, "value": value, "positive": positive,
                             "said": said, "split": split})  # fmt: skip
    out = []
    for key in dict.fromkeys((r["model"], r["group"], r["value"]) for r in rows):
        mine = [r for r in rows if (r["model"], r["group"], r["value"]) == key]
        pos = [r["said"] for r in mine if r["positive"]]
        neg = [r["said"] for r in mine if not r["positive"]]
        out.append({"model": key[0], "group": key[1], "value": key[2],
                    "positives": len(pos), "recall": round(np.mean(pos), 3) if pos else "",
                    "negatives": len(neg), "false_alarms": round(np.mean(neg), 3) if neg else ""})  # fmt: skip
    return out


def within_jc(data, names: list[str]) -> None:
    for name in names:
        line = []
        for split in ("validation", "test"):
            s = {v: sc for (sp, v), sc in scores(name).items() if sp == split}
            videos = list(s)
            truth = np.array([data["category"][list(data["video"]).index(v)] == baseline.POSITIVE
                              for v in videos])  # fmt: skip
            jc = np.array([not v.startswith("pro") for v in videos])
            seeds = np.array([s[v] for v in videos]).T  # (seeds, videos)
            every = np.mean([baseline.auroc(truth, row) for row in seeds])
            only = np.mean([baseline.auroc(truth[jc], row[jc]) for row in seeds])
            line.append(f"{split} {every:.3f} all, {only:.3f} JC only "
                        f"({truth[jc].sum()} drop / {(~truth[jc]).sum()} no)")  # fmt: skip
        print(f"  {name.upper()}: " + "; ".join(line))


def unsee(x: np.ndarray, mask: np.ndarray, share: float, rng: np.random.Generator):
    """Copies with `share` of the seen arm points set unseen, positions interpolated in time."""
    x, mask = x.copy(), mask.copy()
    frames = np.arange(x.shape[1])
    for w in range(len(x)):
        arm = mask[w][:, ARM_JOINTS]
        seen = np.argwhere(arm)
        for t, j in seen[rng.random(len(seen)) < share]:
            mask[w, t, ARM_JOINTS[j]] = False
        for j in ARM_JOINTS:
            known = mask[w, :, j]
            if known.any():
                for c in range(2):
                    x[w, ~known, j, c] = np.interp(frames[~known], frames[known], x[w, known, j, c])
    return x, mask


def predict(name: str, loaded: dict, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean probability over seeds (B0: its drop time) per window."""
    if name == "b0":
        return np.array([baseline.drop_time(w) for w in x])
    return model.probabilities(loaded[name], *model.inputs(loaded[name], x, mask)).mean(0)


def main() -> None:
    data = np.load(dataset.OUT)
    cuts = thresholds()
    loaded = {n: model.load(n) for n in MODELS if n != "b0"}
    table = by_group(data, MODELS, cuts)
    baseline.save("bias", table)
    print("1. Recall and false alarms by group (validation and test scores, mean over seeds):")
    for r in table:
        print(f"  {r['model'].upper():4} {r['group']:16} {r['value']:13} recall {r['recall']!s:5} "
              f"of {r['positives']:2}, false alarms {r['false_alarms']!s:5} of {r['negatives']:2}")  # fmt: skip
    print("2. AUROC on every serve and on JC's serves only:")
    within_jc(data, MODELS)

    test = (data["valid"] == 1) & (data["split"] == "test")
    x, mask = data["x"][test], data["mask"][test]
    rng = np.random.default_rng(0)
    curves = {}
    print("3. Test windows with a share of the arm points unseen (mean probability, share 'drops "
          "early'):")  # fmt: skip
    for name in MODELS:
        curve = []
        for share in SHARES:
            runs = [
                predict(name, loaded, *unsee(x, mask, share, rng))
                for _ in range(REPEATS if share else 1)
            ]
            p = np.mean(runs, 0)
            curve.append((p.mean(), np.mean([(r >= cuts[name]).mean() for r in runs])))
        curves[name] = curve
        print(f"  {name.upper()}: " + ", ".join(f"{s:.0%} unseen {m:.3f} / {d:.0%}"
                                                for s, (m, d) in zip(SHARES, curve)))  # fmt: skip

    mirrored_x, mirrored_mask = x.copy(), mask.copy()
    train.mirror(mirrored_x, mirrored_mask, np.ones(len(x), bool))
    print("4. Mirrored test windows (a left-handed player as a right-handed one), largest change:")
    for name in MODELS:
        change = np.abs(
            predict(name, loaded, mirrored_x, mirrored_mask) - predict(name, loaded, x, mask)
        )
        print(f"  {name.upper()}: {change.max():.2g}")

    front = {}
    if pretrain.THETIS.exists():
        thetis = np.load(pretrain.THETIS)
        expert = np.array([int(p[1:]) >= 32 for p in thetis["player"]])
        print(f"5. THETIS, filmed from the front ({(~expert).sum()} beginner and {expert.sum()} "
              f"expert serves): share called 'drops early' (JC test serves: 13 of 16 have it)")  # fmt: skip
        side = model.shoulder_width(thetis["x"]) >= model.MIN_SHOULDERS
        for name in MODELS:
            said = predict(name, loaded, thetis["x"], thetis["mask"]) >= cuts[name]
            front[name] = (said[~expert].mean(), said[expert].mean())
            print(f"  {name.upper()}: beginners {front[name][0]:.0%}, experts {front[name][1]:.0%}; "
                  f"of the {side.sum()} the app does not send back, {said[side].mean():.0%}")  # fmt: skip
        print(f"  The app asks for a new recording of {1 - side.mean():.0%} of them (shoulders "
              f"under {model.MIN_SHOULDERS} torso lengths apart in the stance)")  # fmt: skip
    plot(curves, front)


def plot(curves: dict, front: dict) -> None:
    _, axes = plt.subplots(1, 2 if front else 1, figsize=(10 if front else 5, 3.6), squeeze=False)
    left = axes[0, 0]
    for name, curve in curves.items():
        left.plot(
            [s * 100 for s in SHARES], [d * 100 for _, d in curve], marker="o", label=name.upper()
        )
    left.set(xlabel="arm points set unseen (%)", ylabel="test serves called 'drops early' (%)",
             title="Fewer points seen", ylim=(0, 105))  # fmt: skip
    left.legend()
    if front:
        right, width = axes[0, 1], 0.38
        names = list(front)
        for k, (label, offset) in enumerate((("beginners", -width / 2), ("experts", width / 2))):
            right.bar(np.arange(len(names)) + offset, [front[n][k] * 100 for n in names], width,
                      label=label)  # fmt: skip
        right.set_xticks(range(len(names)), [n.upper() for n in names])
        right.set(ylabel="THETIS serves called 'drops early' (%)", ylim=(0, 105),
                  title="Camera from the front (THETIS)")  # fmt: skip
        right.legend()
    plt.tight_layout()
    plt.savefig(baseline.RESULTS / "bias.png", dpi=150)


if __name__ == "__main__":
    main()
