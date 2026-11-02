"""Can the landing frame be placed by code? unbalanced_finish, asked as the system asks it (the
two questions of v16), on frames a fixed time after the contact frame of the v16 runs, without
running the whole system.

    uv run python -m experiments.probe_landing [seconds...]

In the v16 runs the landing frame came 2 to 20 source frames after contact on most videos,
with the player still in the air: unbalanced_finish was unclear on 17 of 21 JC serves and 10 of
11 professional ones. No player in the evaluation has that problem (eval/serve/claude_labels.csv),
so "yes" is right, "no" is a false alarm and "unclear" says nothing. Counts per offset, group
and half (dev and test as in experiments/probe_toss_arm.py). Frames saved in
data/processed/probe/landing/.
"""

import collections
import json
import sys
from string import Template

import cv2

from core import ROOT, load_config, video
from core.analysis import FRAME_SIDE, crop, resize
from core.llm import VLM
from experiments.probe_toss_arm import DEV
from strokes import base

RUNS = ["20261102-020925-full", "20261102-021405-full"]
OFFSETS = (0.3, 0.45, 0.6)  # seconds after contact
CONFIG = load_config()
GENERATION = {k: CONFIG["llm"][k] for k in ("temperature", "top_p", "max_tokens", "seed")}


def ask(vlm: VLM, prompts: dict, criteria: dict, image) -> str:
    """unbalanced_finish as checked_feedback asks it: "yes" or "no" when both questions agree."""
    spec, checks = criteria["categories"]["unbalanced_finish"], prompts["checks"]
    phase, votes = spec["phase"], set()
    for field, yes_means in checks["ask"].items():
        user = Template(checks["user"]).substitute(
            phase=phase, looks_like=criteria["phases"][phase]["looks_like"], question=spec[field],
            part="", good="yes", bad="no",
        )  # fmt: skip
        answer = vlm.generate_json(checks["system"], user, [image], base.check_schema(),
                                   **GENERATION).data["answer"]  # fmt: skip
        if yes_means == "bad":
            answer = {"yes": "no", "no": "yes"}.get(answer, answer)
        votes.add(answer)
    return votes.pop() if len(votes) == 1 else "unclear"


def main(seconds: list[float]) -> None:
    vlm, stroke = VLM(CONFIG["llm"]["model"]), base.get("serve")
    prompts, criteria = base.load(stroke, "prompts/v16.yaml"), base.load(stroke, "criteria.yaml")
    out = ROOT / "data" / "processed" / "probe" / "landing"  # players' frames: not in Git
    out.mkdir(parents=True, exist_ok=True)
    tally = collections.defaultdict(collections.Counter)
    for run in RUNS:
        for line in (ROOT / "eval" / "serve" / "runs" / f"{run}.jsonl").open():
            record = json.loads(line)
            if not record.get("key_frames"):
                continue
            path, contact = ROOT / record["video"], record["key_frames"]["contact"]
            fps = video.probe(path).fps
            wanted = {contact + round(s * fps): s for s in seconds}
            frames = {i: f for i, f in video.iter_frames(path) if i in wanted}
            group = "pro" if path.stem.startswith("pro") else "jc"
            half = "dev" if path.stem in DEV else "test"
            row = {"video": path.stem, "fps": round(fps)}
            for index, s in sorted(wanted.items()):
                if index not in frames:  # the video ends before
                    row[str(s)] = "ended"
                    tally[f"+{s}s {group} {half}"]["ended"] += 1
                    continue
                image = resize(crop(frames[index], record.get("box")), FRAME_SIDE)
                cv2.imwrite(str(out / f"{path.stem}.{s}.jpg"), image)
                row[str(s)] = ask(vlm, prompts, criteria, image)
                tally[f"+{s}s {group} {half}"][row[str(s)]] += 1
            print(json.dumps(row), flush=True)
    for key, counts in sorted(tally.items()):
        print(key, dict(counts))


if __name__ == "__main__":
    main([float(s) for s in sys.argv[1:]] or list(OFFSETS))
