"""Evaluate one system variant against the reference labels.

    uv run python -m eval.run_eval --variant full     # key frames + criteria + drills
    uv run python -m eval.run_eval --variant simple   # baseline: same frames, simple prompt
    uv run python -m eval.run_eval --model mlx-community/Qwen3-VL-2B-Instruct-4bit  # baseline
    uv run python -m eval.run_eval --agreement eval/serve/labels_ernesto.csv eval/serve/labels_hugo.csv
    uv run python -m eval.run_eval --videos data/raw/jc-s*.mp4 --note "..."  # no labels yet
    uv run python -m eval.run_eval --review eval/serve/runs/<run>.jsonl eval/serve/review_*.csv
    uv run python -m eval.run_eval --score eval/serve/runs/<run>.jsonl --labels eval/serve/claude_labels.csv

Reads eval/<stroke>/labels.csv (format and labeling guide in data/README.md) and the videos in
data/raw/. --agreement compares the two annotators' independent labels before they are merged.
--videos runs on videos without labels: only the answers and the operational metrics.
--review reads the members' review of a run (made with `python -m eval.label`) and fills its
hallucination rate in experiments/iterations.csv. --score computes the quality metrics of saved
runs against a label file.
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


def review(run_file: Path, review_files: list[Path]) -> None:
    """Hallucination rate of a run from the members' review: the share of checks whose "seen"
    sentence describes something the frame does not show. Also prints how many answers were
    right, per category. Writes the rate in the run's row of experiments/iterations.csv."""
    run = run_file.stem
    rows = [r for p in review_files for r in csv.DictReader(p.read_text().splitlines())
            if r["run"] == run]  # fmt: skip
    if not rows:
        raise SystemExit(f"No review of {run} in the given files.")

    def share(values, wanted, among=("yes", "no")):
        values = [v for v in values if v in among]
        return (round(values.count(wanted) / len(values), 3) if values else float("nan")), len(
            values
        )

    rate, judged = share([r["seen_ok"] for r in rows], "no")
    right, answers = share([r["answer_ok"] for r in rows], "yes")
    print(f"{run}: hallucination rate {rate} ({judged} sentences judged), "
          f"answers right {right} ({answers} judged)")  # fmt: skip
    for category in sorted({r["category"] for r in rows}):
        mine = [r for r in rows if r["category"] == category]
        print(f"- {category}: hallucination {share([r['seen_ok'] for r in mine], 'no')[0]}, "
              f"right {share([r['answer_ok'] for r in mine], 'yes')[0]}")  # fmt: skip

    def fill(row):
        row[10] = str(rate)
        row[15] = f"{row[15]} reviewed: {judged} sentences, answers_right={right}".strip()

    edit_iterations({run: fill})


def score(run_files: list[Path], labels: Path) -> None:
    """Quality metrics of saved runs against a label file, without running the model again:
    for runs made with --videos before the labels existed. Fills them in each run's row of
    experiments/iterations.csv."""
    with open(labels, newline="") as file:
        truth = {r["video"]: r for r in csv.DictReader(file)}
    for label in truth.values():
        label["valid"] = label["valid"].strip() == "1"
    edits, everything = {}, []

    def show(name, m):
        print(f"{name}: n={m['n']} retake_acc={m['retake_acc']} phase_acc={m['phase_acc']} "
              f"P={m['category_precision']} R={m['category_recall']} F1={m['category_f1']} "
              f"drill_acc={m['drill_acc']}")  # fmt: skip

    for run_file in run_files:
        records = [json.loads(line) for line in run_file.open()]
        for record in records:
            record["label"] = truth.get(Path(record["video"]).name)
        records = [r for r in records if r["label"]]
        everything += records
        m = compute(records)
        show(run_file.stem, m)

        def fill(row, m=m):
            row[8], row[9], row[11] = (
                str(m[k]) for k in ("phase_acc", "category_f1", "retake_acc")
            )
            row[15] = (f"{row[15]} scored against {labels.name}: n={m['n']} "
                       f"P={m['category_precision']} R={m['category_recall']} "
                       f"drill_acc={m['drill_acc']}")  # fmt: skip

        edits[run_file.stem] = fill
    if len(run_files) > 1:  # one system's runs, e.g. JC and professionals
        show("together", compute(everything))
    edit_iterations(edits)


def edit_iterations(edits: dict) -> None:
    """Apply edits[run](row) to the rows of experiments/iterations.csv, in place."""
    with open(ITERATIONS, newline="") as file:
        log = list(csv.reader(file))
    for row in log:
        if row[0] in edits:
            edits[row[0]](row)
    with open(ITERATIONS, "w", newline="") as file:
        csv.writer(file, lineterminator="\n").writerows(log)


def compute(records: list[dict]) -> dict[str, float]:
    """Quality and operational metrics. A video wrongly rejected counts as a wrong answer.

    hallucination_rate is an automatic proxy: answers whose category belongs to another phase.
    It is nan when the code chose the category (checks, prompt v6 on): those answers are
    consistent by construction; --review then fills it from the members' review.
    """

    def predicted(record, key):
        return (record.get("prediction") or {}).get(key, "none")

    labeled = [r for r in records if r["label"]]  # --videos: none, quality metrics are nan
    valid = [r for r in labeled if r["label"]["valid"]]
    answered = [r for r in records if r.get("prediction")]
    with_drill = [r for r in valid if r["label"]["drill_id"]]
    precision, recall, f1 = macro_prf(
        [r["label"]["category"] for r in valid], [predicted(r, "category") for r in valid]
    )

    def mean(values):
        return round(float(np.mean(values)), 3) if len(values) else float("nan")

    return {
        "n": len(records),
        "answered": len(answered),
        "flagged": sum(r["prediction"]["category"] != "no_clear_problem" for r in answered),
        "retake_acc": mean([r["valid"] == r["label"]["valid"] for r in labeled]),
        "phase_acc": mean([predicted(r, "phase") == r["label"]["phase"] for r in valid]),
        "category_precision": round(precision, 3),
        "category_recall": round(recall, 3),
        "category_f1": round(f1, 3),
        "drill_acc": mean([predicted(r, "drill_id") == r["label"]["drill_id"] for r in with_drill]),
        "hallucination_rate": mean(
            [not r["prediction"]["consistent"] for r in answered if "checks" not in r["prediction"]]
        ),
        "latency_s": mean([r["seconds"] for r in answered]),
        "tokens": mean([r["tokens"] for r in answered]),
        "mem_gb": max((r["mem_gb"] for r in answered), default=float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=["full", "simple"], default="full")
    parser.add_argument("--model", help="default: llm.model in configs/default.yaml")
    parser.add_argument("--prompt", help="default: llm.prompt_version in configs/default.yaml")
    parser.add_argument("--stroke", default="serve")
    parser.add_argument("--labels", type=Path, help="default: eval/<stroke>/labels.csv")
    parser.add_argument("--note", default="", help="what changed in this iteration")
    parser.add_argument("--videos", nargs="+", type=Path,
                        help="run on these videos without labels (no quality metrics)")  # fmt: skip
    parser.add_argument("--agreement", nargs=2, type=Path, metavar=("A", "B"),
                        help="compare two annotators' label files and exit")  # fmt: skip
    parser.add_argument("--review", nargs="+", type=Path, metavar=("RUN", "REVIEW"),
                        help="hallucination rate of a run from review files, and exit")  # fmt: skip
    parser.add_argument("--score", nargs="+", type=Path, metavar="RUN",
                        help="metrics of saved runs against --labels, and exit")  # fmt: skip
    args = parser.parse_args()
    if args.agreement:
        return agreement(*args.agreement)
    if args.review:
        return review(args.review[0], args.review[1:])
    if args.score:
        return score(args.score, args.labels or ROOT / "eval" / args.stroke / "labels.csv")

    config = load_config()
    stroke = base.get(args.stroke)
    if args.videos:
        rows = [{"video": str(path), "player": "", "path": path, "label": None}
                for path in args.videos]  # fmt: skip
    else:
        labels = args.labels or ROOT / "eval" / args.stroke / "labels.csv"
        with open(labels, newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise SystemExit(
                f"{labels} has no rows yet: add the agreed labels first (data/README.md)."
            )
        for row in rows:
            row["path"] = ROOT / "data" / "raw" / row["video"]
            row["label"] = {**row, "valid": row["valid"].strip() == "1"}

    llm = {**config["llm"], "model": args.model or config["llm"]["model"],
           "prompt_version": args.prompt or config["llm"]["prompt_version"]}  # fmt: skip
    version = llm["prompt_version"]
    generation = {k: llm[k] for k in ("temperature", "top_p", "max_tokens", "seed")}
    vlm = VLM(llm["model"])

    records = []
    for row in rows:
        record = {"video": row["video"], "player": row["player"],
                  "label": row["label"] and {k: v for k, v in row["label"].items() if k != "path"}}  # fmt: skip
        start = time.perf_counter()
        try:
            frames = sample(row["path"], **config["analysis"])
            frames, key_frames, calls = base.phase_frames(stroke, frames, vlm, version,
                                                          **generation)  # fmt: skip
            record["valid"], record["box"] = True, frames.box  # the crop, for eval.label
            record["key_frames"] = {p: frames.indices[i] for p, i in key_frames.items()}  # source
            images = {phase: frames.images[i] for phase, i in key_frames.items()}
            prediction = base.feedback(stroke, images, vlm, args.variant, version, **generation)
            record["prediction"] = prediction
            record["tokens"] = (sum(c.prompt_tokens + c.generation_tokens for c in calls)
                                + prediction["prompt_tokens"] + prediction["generation_tokens"])  # fmt: skip
            record["mem_gb"] = max(*(c.peak_memory_gb for c in calls), prediction["peak_memory_gb"])
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
        csv.writer(file, lineterminator="\n").writerow([
            run, f"{datetime.now():%Y-%m-%d}", args.stroke, f"{args.variant}: {args.note}".strip(": "),
            config["analysis"]["frames"], llm["model"], version, llm["temperature"],
            metrics["phase_acc"], metrics["category_f1"], metrics["hallucination_rate"],
            metrics["retake_acc"], metrics["latency_s"], metrics["tokens"], metrics["mem_gb"],
            (f"n={metrics['n']} answered={metrics['answered']} flagged={metrics['flagged']} "
             f"P={metrics['category_precision']} R={metrics['category_recall']} "
             f"drill_acc={metrics['drill_acc']}"),
        ])  # fmt: skip
    print(json.dumps(metrics, indent=1))
    print(f"Predictions: {out.relative_to(ROOT)}; metrics appended to experiments/iterations.csv")


if __name__ == "__main__":
    main()
