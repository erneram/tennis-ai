"""Evaluate one system variant against the reference labels.

    uv run python -m eval.run_eval --variant full     # key frames + criteria + drills
    uv run python -m eval.run_eval --variant simple   # baseline: same frames, simple prompt
    uv run python -m eval.run_eval --model mlx-community/Qwen3-VL-2B-Instruct-4bit  # baseline
    uv run python -m eval.run_eval --agreement eval/serve/labels_ernesto.csv eval/serve/labels_hugo.csv

Reads eval/<stroke>/labels.csv (format and labeling guide in data/README.md) and the videos in
data/raw/. --agreement compares the two annotators' independent labels before they are merged.
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
from core.analysis import InsufficientEvidence, sample
from core.llm import VLM
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


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two annotators: 1 = same labels, 0 = no better than chance."""
    if not a:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    expected = sum(a.count(c) * b.count(c) for c in set(a) | set(b)) / len(a) ** 2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(path_a: Path, path_b: Path) -> None:
    """Print kappa per column and the videos where the two annotators disagree."""
    a, b = ({r["video"]: r for r in csv.DictReader(p.read_text().splitlines())}
            for p in (path_a, path_b))  # fmt: skip
    videos = sorted(a.keys() & b.keys())
    if not videos:
        raise SystemExit("The two label files share no video.")
    columns = ("valid", "phase", "category", "drill_id")
    for column in columns:
        value = kappa([a[v][column] for v in videos], [b[v][column] for v in videos])
        print(f"{column}: kappa {value:.2f}")
    print(f"{len(videos)} videos in both files. Disagreements to settle together:")
    for v in videos:
        if any(a[v][c] != b[v][c] for c in columns[:3]):
            x, y = a[v], b[v]
            print(f"- {v}: {x['phase']}/{x['category']} vs {y['phase']}/{y['category']}")


def compute(records: list[dict]) -> dict[str, float]:
    """Quality and operational metrics. A video wrongly rejected counts as a wrong answer.

    hallucination_rate is an automatic proxy: answers whose category belongs to another phase.
    Whether each observation matches its frames is checked by hand in the written report.
    """

    def predicted(record, key):
        return (record.get("prediction") or {}).get(key, "none")

    valid = [r for r in records if r["label"]["valid"]]
    answered = [r for r in records if r.get("prediction")]
    with_drill = [r for r in valid if r["label"]["drill_id"]]
    precision, recall, f1 = macro_prf(
        [r["label"]["category"] for r in valid], [predicted(r, "category") for r in valid]
    )

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
        "hallucination_rate": mean([not r["prediction"]["consistent"] for r in answered]),
        "latency_s": mean([r["seconds"] for r in answered]),
        "tokens": mean([r["tokens"] for r in answered]),
        "mem_gb": max((r["mem_gb"] for r in answered), default=float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=["full", "simple"], default="full")
    parser.add_argument("--model", help="default: llm.model in configs/default.yaml")
    parser.add_argument("--stroke", default="serve")
    parser.add_argument("--labels", type=Path, help="default: eval/<stroke>/labels.csv")
    parser.add_argument("--note", default="", help="what changed in this iteration")
    parser.add_argument("--agreement", nargs=2, type=Path, metavar=("A", "B"),
                        help="compare two annotators' label files and exit")  # fmt: skip
    args = parser.parse_args()
    if args.agreement:
        return agreement(*args.agreement)

    config = load_config()
    stroke = base.get(args.stroke)
    labels = args.labels or ROOT / "eval" / args.stroke / "labels.csv"
    with open(labels, newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"{labels} has no rows yet: add the agreed labels first (data/README.md).")

    llm = {**config["llm"], "model": args.model or config["llm"]["model"]}
    version = llm["prompt_version"]
    generation = {k: llm[k] for k in ("temperature", "top_p", "max_tokens", "seed")}
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
            frames = sample(ROOT / "data" / "raw" / row["video"], **config["analysis"])
            key_frames, located = base.locate_phases(stroke, frames.images, vlm, version,
                                                     **generation)  # fmt: skip
            record["valid"], record["key_frames"] = True, key_frames
            images = {phase: frames.images[i] for phase, i in key_frames.items()}
            prediction = base.feedback(stroke, images, vlm, args.variant, version, **generation)
            record["prediction"] = prediction
            record["tokens"] = (located.prompt_tokens + located.generation_tokens
                                + prediction["prompt_tokens"] + prediction["generation_tokens"])  # fmt: skip
            record["mem_gb"] = max(located.peak_memory_gb, prediction["peak_memory_gb"])
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

    with open(ITERATIONS, "a", newline="") as file:
        csv.writer(file).writerow([
            run, f"{datetime.now():%Y-%m-%d}", args.stroke, f"{args.variant}: {args.note}".strip(": "),
            config["analysis"]["frames"], llm["model"], version, llm["temperature"],
            metrics["phase_acc"], metrics["category_f1"], metrics["hallucination_rate"],
            metrics["retake_acc"], metrics["latency_s"], metrics["tokens"], metrics["mem_gb"],
            (f"n={metrics['n']} P={metrics['category_precision']} "
             f"R={metrics['category_recall']} drill_acc={metrics['drill_acc']}"),
        ])  # fmt: skip
    print(json.dumps(metrics, indent=1))
    print(f"Predictions: {out.relative_to(ROOT)}; metrics appended to experiments/iterations.csv")


if __name__ == "__main__":
    main()
