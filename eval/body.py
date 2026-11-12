"""How far the team's answers about the body can be trusted, with no second annotator.

    uv run python -m eval.body [--name ernesto]

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
"""

import argparse

import numpy as np

from core import ROOT
from eval.run_eval import kappa
from strokes.serve import baseline, body, dataset

EVAL = ROOT / "eval" / "serve"
MIN_CLASS = 5
MIN_KAPPA = 0.40


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


if __name__ == "__main__":
    main()
