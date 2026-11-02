"""How much do the results depend on the sampling seed? Macro-F1 per prompt version and seed,
over the full runs of JC and the professionals, against eval/serve/claude_labels.csv.

    uv run python -m experiments.seeds

Runs are found by their note in experiments/iterations.csv ("vNN, seed N, uncropped JC videos"
and "vNN, seed N, professionals", from eval/run_eval.py --seed); seed 0 runs are listed below.
Per seed: macro-F1 in total and per half (dev and test as in experiments/probe_toss_arm.py), JC
serves flagged for the tossing arm, professionals flagged for anything (false alarms), and
valid videos with no answer (errors).

Result (2026-11-02), macro-F1 for seeds 0 to 3:
- v12: 0.280, 0.220, 0.175, 0.165; mean 0.210 ± 0.052.
- v17: 0.574, 0.206, 0.329, 0.233; mean 0.336 ± 0.168; errors 0, 2, 3, 1.
- v18: 0.574, 0.343, 0.337, 0.233; mean 0.372 ± 0.144; errors 0, 0, 1, 1.
- v19: 0.366, 0.412, 0.264, 0.233; mean 0.319 ± 0.084; errors 0, 0, 1, 1.
On seed 1 the contact frame moved 1 to 18 source frames from seed 0, and the tossing-arm answer
flipped on some identical frames: one seed is not enough to compare two versions.
"""

import csv
import json
import re
import statistics
from pathlib import Path

from core import ROOT
from eval.run_eval import compute
from experiments.probe_toss_arm import DEV

SEED_0 = {  # before --seed: the config's seed, 0
    "v12": ["20261102-002057-full", "20261102-002636-full"],
    "v17": ["20261102-023129-full", "20261102-023612-full"],
    # v18 only adds a third try after two guesses; with seed 0 v17 never guessed twice, so
    # v18 makes exactly v17's calls
    "v18": ["20261102-023129-full", "20261102-023612-full"],
}
NOTE = re.compile(r"full: (v\d+), seed (\d+), (uncropped JC videos|professionals)$")


def runs() -> dict[tuple[str, int], list[str]]:
    found = {(version, 0): list(names) for version, names in SEED_0.items()}
    for row in csv.DictReader((ROOT / "experiments" / "iterations.csv").open()):
        match = NOTE.match(row["change"])
        if match and row["vlm_model"].endswith("4B-Instruct-4bit"):
            found.setdefault((match[1], int(match[2])), []).append(row["id"])
    return {key: names for key, names in found.items() if len(names) == 2}


def main() -> None:
    with (ROOT / "eval" / "serve" / "claude_labels.csv").open() as file:
        labels = {row["video"]: row for row in csv.DictReader(file)}
    for row in labels.values():
        row["valid"] = row["valid"].strip() == "1"
    scores = {}
    for (version, seed), names in sorted(runs().items()):
        records = []
        for name in names:
            for line in (ROOT / "eval" / "serve" / "runs" / f"{name}.jsonl").open():
                record = json.loads(line)
                record["label"] = labels.get(Path(record["video"]).name)
                if record["label"]:
                    records.append(record)
        found = [
            (Path(r["video"]).stem, (r.get("prediction") or {}).get("category")) for r in records
        ]
        arm = sum(c == "toss_arm_drops_early" for v, c in found if not v.startswith("pro"))
        alarms = sum(c not in (None, "no_clear_problem") for v, c in found if v.startswith("pro"))
        errors = sum(not r.get("prediction") and r["label"]["valid"] for r in records)
        f1 = compute(records)["category_f1"]
        dev = compute([r for r in records if Path(r["video"]).stem in DEV])["category_f1"]
        test = compute([r for r in records if Path(r["video"]).stem not in DEV])["category_f1"]
        scores.setdefault(version, []).append(f1)
        print(f"{version} seed {seed}: macro-F1 {f1} (dev {dev}, test {test}), JC tossing arm "
              f"{arm}, professional false alarms {alarms}, errors {errors}")  # fmt: skip
    for version, f1s in scores.items():
        spread = f" ± {statistics.stdev(f1s):.3f}" if len(f1s) > 1 else ""
        print(f"{version}: {len(f1s)} seeds, macro-F1 {statistics.mean(f1s):.3f}{spread}")


if __name__ == "__main__":
    main()
