"""Value model of the Final Project (written report, section 4.3): what each serve model is worth
to an amateur who practices alone, in quetzales.

    uv run python -m eval.value

Every analysis ends in one drill block: toss_hold when the model finds that the tossing arm drops
early, steady serves when it does not. With p the share of amateur serves that have the fault,
TPR and FPR the model's rates at its threshold and B the price of one drill block with a coach,
the value of one analysis is

    B × (p × TPR × hit − p × (1 − TPR) × miss − (1 − p) × FPR × false_alarm) − compute cost

and a player makes analyses_per_month of them. Parameters and their sources: configs/value.yaml.
Models: B0, M1, M1b, M1c, M1d, M2, M2c (experiments/serve/<model>.csv, from strokes.serve.baseline and
.train)
and Qwen3-VL, the Project 3 system, from its runs. Each model's threshold is the one with the
most value on its validation scores, never on the test split; test values get a 95 % bootstrap
interval over the test videos. Writes experiments/serve/value.csv, sensitivity.csv (every
parameter at ±20 % and ±50 %, thresholds chosen again) and value-curve.png (value against recall).
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from core import ROOT
from strokes.serve import baseline, dataset

CONFIG = ROOT / "configs" / "value.yaml"
RUNS = ROOT / "eval" / "serve" / "runs"
# Qwen3-VL with prompt v22, the final Project 3 version (eval/label.py, REVIEW_RUNS)
QWEN_RUNS = {"validation": ["20261102-131618-full", "20261102-132154-full"],
             "test": ["20261106-133044-full"]}  # fmt: skip
MODELS = ["b0", "m1", "m1b", "m1c", "m1d", "m1e", "m1f", "m2", "m2c", "qwen"]
FACTORS = [0.5, 0.8, 1.2, 1.5]
BOOTSTRAP = 2000


def compute_cost(config: dict, seconds: float) -> float:
    """Quetzales of electricity for one analysis."""
    return seconds / 3600 * config["watts"] / 1000 * config["kwh_price"]


def value(tpr, fpr, config: dict, seconds: float):
    """Quetzales per analysis (the formula of the docstring); tpr and fpr may be arrays."""
    c = config
    block = c["coach_per_hour"] * c["drill_minutes"] / 60
    p = min(c["prevalence"], 1)
    gain = p * tpr * c["hit"] - p * (1 - tpr) * c["miss"] - (1 - p) * fpr * c["false_alarm"]
    return block * gain - compute_cost(c, seconds)


def rates(truth: np.ndarray, scores: np.ndarray, threshold: float):
    """TPR and FPR per seed; scores is (seeds, videos), positive when score >= threshold."""
    said = scores >= threshold
    return said[:, truth].mean(1), said[:, ~truth].mean(1)


def mean_value(truth, scores, threshold, config, seconds) -> float:
    return float(np.mean(value(*rates(truth, scores, threshold), config, seconds)))


def best_threshold(truth, scores, config, seconds) -> float:
    """The validation threshold with the most value (np.inf: never say "drops early")."""
    candidates = [*np.unique(scores), np.inf]
    return max(candidates, key=lambda t: mean_value(truth, scores, t, config, seconds))


def from_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Truth per video and (seeds, videos) scores from rows of strokes.serve.baseline.COLUMNS."""
    videos = sorted({r["video"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    score = {(r["seed"], r["video"]): float(r["score"]) for r in rows}
    truth = {r["video"]: r["truth"] == baseline.POSITIVE for r in rows}
    return np.array([truth[v] for v in videos]), np.array([[score[s, v] for v in videos]
                                                          for s in seeds])  # fmt: skip


def qwen_rows() -> tuple[list[dict], float]:
    """Rows like the other models' from the Qwen runs, and its mean seconds per test video.
    The check answers "yes" when the arm is fine: "no" scores 1, "unclear" 0.5, and a video it
    did not judge (asked for another recording, or failed) 0, since no drill follows."""
    data = np.load(dataset.OUT)
    labeled = {v: (s, c) for v, s, c, ok in
               zip(data["video"], data["split"], data["category"], data["valid"]) if ok == 1}  # fmt: skip
    rows, seconds = [], []
    for split, runs in QWEN_RUNS.items():
        for run in runs:
            for line in (RUNS / f"{run}.jsonl").open():
                record = json.loads(line)
                video = Path(record["video"]).stem
                wanted = "train" if split == "validation" else "test"
                if labeled.get(video, ("",))[0] != wanted:
                    continue
                check = (record.get("prediction") or {}).get("checks", {}).get(
                    baseline.POSITIVE, {})  # fmt: skip
                score = {"no": 1.0, "unclear": 0.5}.get(check.get("answer"), 0.0)
                rows.append({"seed": 0, "split": split, "video": video, "score": score,
                             "truth": labeled[video][1]})  # fmt: skip
                if split == "test":
                    seconds.append(record["seconds"])
    return rows, float(np.mean(seconds)) if seconds else float("nan")


def load(config: dict) -> dict[str, dict]:
    """Per model: its validation and test (truth, scores) and its seconds per analysis."""
    models = {}
    for name in MODELS:
        if name == "qwen":
            rows, seconds = qwen_rows()
        else:
            path = baseline.RESULTS / f"{name}.csv"
            if not path.exists():
                continue
            rows, seconds = list(csv.DictReader(path.open())), config["pose_seconds"]
        splits = {s: from_rows([r for r in rows if r["split"] == s]) for s in ("validation", "test")
                  if any(r["split"] == s for r in rows)}  # fmt: skip
        if len(splits) == 2:
            models[name] = {**splits, "seconds": seconds}
    return models


def bootstrap(truth, scores, threshold, config, seconds, rng) -> tuple[float, float]:
    """95 % interval of the test value, resampling positive and negative videos separately."""
    positive, negative = np.flatnonzero(truth), np.flatnonzero(~truth)
    values = []
    for _ in range(BOOTSTRAP):
        chosen = np.concatenate([rng.choice(positive, len(positive)),
                                 rng.choice(negative, len(negative))])  # fmt: skip
        values.append(mean_value(truth[chosen], scores[:, chosen], threshold, config, seconds))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def evaluate(models: dict, config: dict) -> list[dict]:
    """Threshold from validation, test rates and value of each model."""
    out = []
    for name, m in models.items():
        threshold = best_threshold(*m["validation"], config, m["seconds"])
        truth, scores = m["test"]
        tpr, fpr = rates(truth, scores, threshold)
        test = mean_value(truth, scores, threshold, config, m["seconds"])
        rng = np.random.default_rng(0)  # the same resamples for every model
        low, high = bootstrap(truth, scores, threshold, config, m["seconds"], rng)
        out.append({
            "model": name, "threshold": threshold, "seconds": round(m["seconds"], 2),
            "validation_value": mean_value(*m["validation"], threshold, config, m["seconds"]),
            "test_tpr": float(tpr.mean()), "test_fpr": float(fpr.mean()), "test_value": test,
            "test_low": low, "test_high": high,
            "per_month": test * config["analyses_per_month"],
            "compute": compute_cost(config, m["seconds"]),
        })  # fmt: skip
    return out


def sensitivity(models: dict, config: dict) -> list[dict]:
    """Test value of each model with one parameter off by each factor, thresholds chosen again
    (no compute parameter matters next to the drill block)."""
    rows = []
    for parameter in [k for k in config if k not in ("pose_seconds", "watts", "kwh_price")]:
        for factor in FACTORS:
            changed = {**config, parameter: config[parameter] * factor}
            for name, m in models.items():
                threshold = best_threshold(*m["validation"], changed, m["seconds"])
                test = mean_value(*m["test"], threshold, changed, m["seconds"])
                rows.append({"parameter": parameter, "factor": factor, "model": name,
                             "threshold": threshold, "test_value": test,
                             "per_month": test * changed["analyses_per_month"]})  # fmt: skip
    return rows


def plot(models: dict, results: list[dict], config: dict) -> None:
    """Test value per analysis against recall as each model's threshold moves."""
    _, ax = plt.subplots(figsize=(6, 3.8))
    for r in results:
        m = models[r["model"]]
        truth, scores = m["test"]
        points = []
        for t in [*np.unique(scores), np.inf]:
            tpr, _ = rates(truth, scores, t)
            points.append((tpr.mean(), mean_value(truth, scores, t, config, m["seconds"])))
        line = ax.plot(*zip(*sorted(points)), marker=".", label=r["model"].upper())[0]
        ax.plot(r["test_tpr"], r["test_value"], "o", ms=9, mfc="none", color=line.get_color())
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="recall of 'drops early' (test)", ylabel="Q per analysis (test)",
           title="Value by threshold; circles: threshold chosen on validation")  # fmt: skip
    ax.legend()
    plt.tight_layout()
    plt.savefig(baseline.RESULTS / "value-curve.png", dpi=150)


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    models = load(config)
    results = evaluate(models, config)
    block = config["coach_per_hour"] * config["drill_minutes"] / 60
    p = config["prevalence"]
    print(f"One drill block with a coach: Q{block:.2f}. Per analysis, +0.10 recall is worth "
          f"Q{0.1 * block * p * (config['hit'] + config['miss']):.2f} and +0.10 false-alarm rate "
          f"costs Q{0.1 * block * (1 - p) * config['false_alarm']:.2f}. Always toss_hold, no "
          f"model: Q{value(1, 1, config, 0):.2f}.")  # fmt: skip
    for r in results:
        print(f"  {r['model'].upper():5} threshold {r['threshold']:.3g}: test recall "
              f"{r['test_tpr']:.2f}, false alarms {r['test_fpr']:.2f}, Q{r['test_value']:.2f} per "
              f"analysis (95 % CI {r['test_low']:.2f} to {r['test_high']:.2f}), "
              f"Q{r['per_month']:.0f} per player per month, {r['seconds']} s")  # fmt: skip
    rows = sensitivity(models, config)
    print("Sensitivity (best model, test Q per player per month):")
    for parameter in dict.fromkeys(r["parameter"] for r in rows):
        cells = []
        for factor in FACTORS:
            group = [r for r in rows if r["parameter"] == parameter and r["factor"] == factor]
            top = max(group, key=lambda r: r["per_month"])
            cells.append(f"x{factor}: {top['model'].upper()} {top['per_month']:.0f}")
        print(f"  {parameter}: " + ", ".join(cells))
    baseline.save("value", results)
    baseline.save("sensitivity", rows)
    plot(models, results, config)


if __name__ == "__main__":
    main()
