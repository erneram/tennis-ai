"""Evaluate one system variant against the coach labels.

    uv run python -m eval.run_eval --variant full     # frames + measurements + coach criteria
    uv run python -m eval.run_eval --variant simple   # baseline: same frames, simple prompt
    uv run python -m eval.run_eval --variant rules    # baseline: fixed rules, no language model

Reads eval/<stroke>/labels.csv (format in data/README.md) and the videos in data/raw/.
Writes one JSON line per video to eval/<stroke>/runs/<run>.jsonl and appends the metrics
to experiments/iterations.csv.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from core import ROOT, load_config
from core.analysis import InsufficientEvidence, analyze
from strokes import base

ITERATIONS = ROOT / "experiments" / "iterations.csv"


def macro_prf(truth: list[str], predicted: list[str]) -> tuple[float, float, float]:
    """Macro-averaged precision, recall and F1 over every label seen in either list."""
    scores = []
    for label in sorted(set(truth) | set(predicted)):
        tp = sum(t == p == label for t, p in zip(truth, predicted))
        n_pred, n_true = predicted.count(label), truth.count(label)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true if n_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append((precision, recall, f1))
    if not scores:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(np.mean(column)) for column in zip(*scores))


def grounded(prediction: dict, criteria: dict, measurements: dict) -> bool:
    """Automatic faithfulness proxy: the category belongs to the chosen phase and, if the category
    has a measurable check, the measurement really is outside the coach range."""
    if criteria["categories"][prediction["category"]]["phase"] != prediction["phase"]:
        return False
    checks = [
        r
        for r in base.check_results(criteria, measurements)
        if r["category"] == prediction["category"]
    ]
    return not checks or any(r["deviation"] > 0 for r in checks)


def compute(records: list[dict]) -> dict[str, float]:
    """Quality and operational metrics. A video wrongly rejected counts as a wrong answer."""

    def predicted(record, key):
        return (record.get("prediction") or {}).get(key, "none")

    valid = [r for r in records if r["label"]["valid"]]
    answered = [r for r in records if r.get("prediction")]
    with_drill = [r for r in valid if r["label"]["drill_id"]]
    precision, recall, f1 = macro_prf(
        [r["label"]["category"] for r in valid], [predicted(r, "category") for r in valid]
    )
    vlm = [r["prediction"] for r in answered if "prompt_tokens" in r["prediction"]]

    def mean(values):
        return round(float(np.mean(values)), 3) if len(values) else float("nan")

    return {
        "n": len(records),
        "retake_acc": mean([r["valid"] == r["label"]["valid"] for r in records]),
        "phase_acc": mean([predicted(r, "phase") == r["label"]["phase"] for r in valid]),
        "category_precision": round(precision, 3),
        "category_recall": round(recall, 3),
        "category_f1": round(f1, 3),
        "drill_acc": mean([predicted(r, "drill_id") == r["label"]["drill_id"] for r in with_drill]),
        "hallucination_rate": mean([not r["grounded"] for r in answered]),
        "latency_s": mean([r["seconds"] for r in answered]),
        "tokens": mean([p["prompt_tokens"] + p["generation_tokens"] for p in vlm]),
        "mem_gb": max((p["peak_memory_gb"] for p in vlm), default=float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=["full", "simple", "rules"], default="full")
    parser.add_argument("--stroke", default="serve")
    parser.add_argument("--labels", type=Path, help="default: eval/<stroke>/labels.csv")
    parser.add_argument("--note", default="", help="what changed in this iteration")
    args = parser.parse_args()

    config = load_config()
    stroke = base.get(args.stroke)
    criteria = base.load(stroke, "criteria.yaml")
    labels = args.labels or ROOT / "eval" / args.stroke / "labels.csv"
    with open(labels, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"{labels} has no rows yet: add the coach labels first.")

    pose, llm = config["pose"], config["llm"]
    generation = {k: llm[k] for k in ("temperature", "top_p", "max_tokens", "seed")}
    vlm = None
    if args.variant != "rules":
        from core.llm import VLM

        vlm = VLM(llm["model"])

    records = []
    for row in rows:
        record = {
            "video": row["video"],
            "player": row["player"],
            "label": {**row, "valid": row["valid"].strip() == "1"},
        }
        start = time.perf_counter()
        try:
            analysis = analyze(ROOT / "data" / "raw" / row["video"], stroke, pose["backend"],
                               pose[pose["backend"]], **config["analysis"])  # fmt: skip
            record["valid"], record["measurements"] = True, analysis.measurements
            if args.variant == "rules":
                prediction = base.rule_feedback(stroke, analysis.measurements)
            else:
                prediction = base.feedback(stroke, analysis, vlm, args.variant,
                                           llm["prompt_version"], **generation)  # fmt: skip
            record["prediction"] = prediction
            record["grounded"] = grounded(prediction, criteria, analysis.measurements)
        except InsufficientEvidence as reason:
            record["valid"], record["reason"] = False, str(reason)
        except Exception as error:  # noqa: BLE001 - a failed answer is a wrong answer, not a crash
            record["valid"], record["error"] = True, repr(error)
        record["seconds"] = round(time.perf_counter() - start, 2)
        records.append(record)
        answer = (
            record.get("prediction", {}).get("category")
            or record.get("reason")
            or record.get("error")
        )
        print(f"{row['video']}: {answer}")

    metrics = compute(records)
    run = f"{datetime.now():%Y%m%d-%H%M%S}-{args.variant}"
    out = ROOT / "eval" / args.stroke / "runs" / f"{run}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records))

    uses_vlm = args.variant != "rules"
    with open(ITERATIONS, "a", newline="") as file:
        csv.writer(file).writerow([
            run, f"{datetime.now():%Y-%m-%d}", args.stroke, f"{args.variant}: {args.note}".strip(": "),
            f"{pose['backend']}:{pose[pose['backend']]}", llm["model"] if uses_vlm else "",
            llm["prompt_version"] if uses_vlm else "", llm["temperature"] if uses_vlm else "",
            metrics["phase_acc"], metrics["category_f1"], metrics["hallucination_rate"],
            metrics["retake_acc"], metrics["latency_s"], metrics["tokens"], metrics["mem_gb"],
            (f"n={metrics['n']} P={metrics['category_precision']} "
             f"R={metrics['category_recall']} drill_acc={metrics['drill_acc']}"),
        ])  # fmt: skip
    print(json.dumps(metrics, indent=1))
    print(f"Predictions: {out.relative_to(ROOT)}; metrics appended to experiments/iterations.csv")


if __name__ == "__main__":
    main()
