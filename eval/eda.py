"""Exploratory analysis of the serve data of the Final Project (written report, section 4.2).

    uv run python -m eval.eda

Reads the team's labels (eval/serve/labels.csv and each rater's file), the videos in data/raw/
and data/processed/serve_dataset.npz. Prints the tables of the report and writes
experiments/serve/eda.csv (one row per video, no image) and eda.png: the tossing wrist's height
over the window and when it goes below the shoulder (B0's measure), by group. JC is the only
amateur, so every serve with the fault is his: the groups keep JC's serves without the fault apart
from the professionals'.
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import ROOT, video
from eval.run_eval import kappa
from strokes.serve import baseline, dataset, train

RAW = ROOT / "data" / "raw"
RATERS = ["ernesto", "hugo"]
GROUPS = {"JC, drops early": ("jc", baseline.POSITIVE), "JC, no clear problem": ("jc", baseline.NEGATIVE),
          "JC after feedback": ("jc-post", baseline.NEGATIVE), "EA": ("ea", baseline.NEGATIVE),
          "professionals": ("pro", baseline.NEGATIVE)}  # fmt: skip


def kind(row: dict) -> str:
    """pro, jc (before the feedback), jc-post (after it, dataset.NEW) or another amateur."""
    if row["player"].startswith("pro"):
        return "pro"
    return "jc-post" if row["video"].startswith("jc-post") else row["player"]


def rows(data) -> list[dict]:
    """One row per video: its group, recording and what MediaPipe and B0 see."""
    arms_mask = train.arms(data["x"], data["mask"])[1]
    out = []
    for i, name in enumerate(data["video"]):
        info = video.probe(next(RAW.glob(f"{name}.*")))
        out.append({
            "video": name, "player": data["player"][i], "split": data["split"][i],
            "valid": int(data["valid"][i]), "category": data["category"][i],
            "fps": round(info.fps, 1), "width": info.width, "height": info.height,
            "seconds": round(info.frame_count / info.fps, 2),
            "contact_s": round(data["contact"][i] / data["video_fps"][i], 2),
            "seen_all": round(float(data["mask"][i].mean()), 3),
            "seen_arms": round(float(arms_mask[i].mean()), 3),
            "tossing_side": baseline.tossing_side(data["x"][i]),
            "drop_seconds": round(baseline.drop_time(data["x"][i]), 3),
        })  # fmt: skip
    return out


def agreement() -> None:
    """Kappa between the two raters before they settled their disagreements."""
    files = [ROOT / "eval" / "serve" / f"labels_{n}.csv" for n in RATERS]
    labels = [{r["video"]: r for r in csv.DictReader(f.read_text().splitlines())} for f in files]
    videos = sorted(set(labels[0]) & set(labels[1]))
    valid = [[labels[k][v]["valid"] for v in videos] for k in (0, 1)]
    both = [v for v in videos if labels[0][v]["valid"] == labels[1][v]["valid"] == "1"]
    category = [[labels[k][v]["category"] for v in both] for k in (0, 1)]
    same = sum(a == b for a, b in zip(*category))
    print(f"Raters ({len(videos)} videos): kappa {kappa(*valid):.2f} on 'can be judged', "
          f"{kappa(*category):.2f} on the category of the {len(both)} both judged "
          f"({same} the same, {len(both) - same} settled together)")  # fmt: skip


def main() -> None:
    data = np.load(dataset.OUT)
    table = rows(data)
    baseline.save("eda", table)
    agreement()
    print("Videos by player and split (valid: drops early / no clear problem; not valid):")
    for player in sorted({r["player"] for r in table}):
        for split in ("train", "test", "new"):
            mine = [r for r in table if r["player"] == player and r["split"] == split]
            if mine:
                drop = sum(r["category"] == baseline.POSITIVE for r in mine)
                none = sum(r["category"] == baseline.NEGATIVE for r in mine)
                print(f"  {player:6} {split:5}: {len(mine):2} videos, {drop:2} / {none:2}; "
                      f"{sum(r['valid'] == 0 for r in mine)} not valid")  # fmt: skip
    print("Recordings and landmarks by player type:")
    for group in ("jc", "jc-post", "ea", "pro"):
        mine = [r for r in table if kind(r) == group]
        fps = sorted({round(r["fps"]) for r in mine})
        sizes = sorted({f"{r['width']}x{r['height']}" for r in mine})
        seconds = np.array([r["seconds"] for r in mine])
        print(f"  {group}: {len(mine)} videos, fps {fps}, sizes {sizes}, {np.median(seconds):.1f} s "
              f"median ({seconds.min():.1f} to {seconds.max():.1f}); MediaPipe saw "
              f"{np.mean([r['seen_all'] for r in mine]):.0%} of all points and "
              f"{np.mean([r['seen_arms'] for r in mine]):.0%} of the arm points; tossing arm "
              f"{ {str(k): int(n) for k, n in zip(*np.unique([r['tossing_side'] for r in mine], return_counts=True))} }")  # fmt: skip
    for valid in (1, 0):
        seen = [r["seen_arms"] for r in table if r["valid"] == valid]
        print(f"  {'valid' if valid else 'not valid'}: arm points seen {min(seen):.2f} to "
              f"{max(seen):.2f}, mean {np.mean(seen):.2f}")  # fmt: skip
    print("Seconds before contact the tossing wrist goes below its shoulder (B0's measure):")
    for name, (group, category) in GROUPS.items():
        drop = np.array([r["drop_seconds"] for r in table
                         if kind(r) == group and r["category"] == category])  # fmt: skip
        low, mid, high = np.percentile(drop, [25, 50, 75])
        print(f"  {name}: median {mid:.2f} s (quartiles {low:.2f} to {high:.2f}), n = {len(drop)}")
    plot(data, table)


def plot(data, table: list[dict]) -> None:
    x = train.arms(data["x"], data["mask"])[0]
    height = x[:, :, 0, 1] - x[:, :, 2, 1]  # tossing shoulder minus wrist: image y grows downward
    seconds = (np.arange(x.shape[1]) - dataset.BEFORE) / dataset.FPS
    threshold = next(float(r["threshold"]) for r in csv.DictReader(
        (baseline.RESULTS / "value.csv").open()) if r["model"] == "b0")  # fmt: skip
    _, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    for n, (name, (group, category)) in enumerate(GROUPS.items()):
        chosen = np.array([kind(r) == group and r["category"] == category for r in table])
        mean, sd = height[chosen].mean(0), height[chosen].std(0)
        line = left.plot(seconds, mean, label=f"{name} ({chosen.sum()})")[0]
        left.fill_between(seconds, mean - sd, mean + sd, color=line.get_color(), alpha=0.15)
        drop = [r["drop_seconds"] for r, c in zip(table, chosen) if c]
        right.scatter(drop, np.full(len(drop), n) + np.linspace(-0.2, 0.2, len(drop)),
                      color=line.get_color(), s=18)  # fmt: skip
    left.axhline(0, color="grey", lw=0.8)
    left.axvline(0, color="grey", lw=0.8, ls=":")
    left.set(xlabel="seconds from contact", ylabel="wrist above shoulder (torso lengths)",
             title="Tossing wrist height, mean ± SD")  # fmt: skip
    left.legend(fontsize=8)
    right.axvline(threshold, color="black", ls="--", lw=1, label=f"B0 threshold {threshold:.2f} s")
    right.set_yticks(range(len(GROUPS)), list(GROUPS))
    right.set(
        xlabel="seconds before contact", title="When the tossing wrist goes below the shoulder"
    )
    right.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(baseline.RESULTS / "eda.png", dpi=150)


if __name__ == "__main__":
    main()
