"""Try other wordings or key frames for a check on saved key frames, without running the whole
system. Source of the probe numbers in the header of strokes/serve/prompts/v10.yaml.

    uv run python -m experiments.probe_checks cache   # boxes and key frames of the v9 runs
    uv run python -m experiments.probe_checks probe   # every probe below, JC vs professionals

The key frames are those the v9 runs chose (and v8's follow-through), decoded again and cropped
as the model saw them, so a probe changes only the question or the frame.
"""

import collections
import json
import sys
from pathlib import Path
from string import Template

import cv2

from core import ROOT, load_config, video
from core.analysis import FRAME_SIDE, crop, resize, sample
from core.llm import VLM
from strokes import base

CACHE = ROOT / "data" / "processed" / "probe"  # crops of players' videos: not in Git
RUNS = ROOT / "eval" / "serve" / "runs"
V9 = ["20261101-214840-full", "20261101-215400-full"]
V8 = ["20261101-220401-full", "20261101-220923-full"]
CONFIG = load_config()
GENERATION = {k: CONFIG["llm"][k] for k in ("temperature", "top_p", "max_tokens", "seed")}
STROKE = base.get("serve")

TOSS_ARM = "the tossing arm (the arm that releases the ball, not the racket arm)"
TOSS_ARM_AFTER = "the tossing arm (the arm that released the ball, not the racket arm)"
# name: (key frame, check question, problem question)
PROBES = {
    "toss_arm_bent, toss frame (v9)": (
        "v9_toss", f"Is {TOSS_ARM} straight or almost straight?",
        f"Is {TOSS_ARM} clearly bent at the elbow?"),
    "toss_arm_bent, trophy frame": (
        "v9_trophy", f"Is {TOSS_ARM} straight or almost straight?",
        f"Is {TOSS_ARM} clearly bent at the elbow?"),
    "toss_arm_bent, trophy frame, new wording (v10)": (
        "v9_trophy",
        f"Is {TOSS_ARM_AFTER} straight or almost straight, reaching up toward the ball?",
        f"Is {TOSS_ARM_AFTER} folded at the elbow, with the hand close to the head?"),
    "unbalanced_finish (v9)": (
        "v8_follow_through", "Is the player balanced, landing on the front foot without stumbling?",
        "Is the player stumbling, leaning over, or landing on the back foot?"),
    "unbalanced_finish, new wording": (
        "v8_follow_through", "Is the player upright and under control, with the body above the feet?",
        "Is the player falling, stumbling, or clearly off balance?"),
}  # fmt: skip


def records(runs: list[str]) -> dict[str, dict]:
    out = {}
    for run in runs:
        for line in (RUNS / f"{run}.jsonl").open():
            record = json.loads(line)
            if record.get("key_frames"):
                out[Path(record["video"]).name] = record
    return out


def cache(vlm: VLM) -> None:
    """The player's box (prompt v9) and the key frames of every video, cropped to it."""
    CACHE.mkdir(parents=True, exist_ok=True)
    v8 = records(V8)
    for name, record in records(V9).items():
        path = ROOT / "data" / "raw" / name
        if (CACHE / f"{name}.json").exists() or not path.exists():  # jc-s27 was split in two
            continue
        frames = sample(path, **CONFIG["analysis"])
        box, _ = base.find_player(STROKE, frames.images, vlm, "v9", **GENERATION)
        wanted = {f"v9_{phase}": index for phase, index in record["key_frames"].items()}
        if name in v8:
            wanted["v8_follow_through"] = v8[name]["key_frames"]["follow_through"]
        for key, index in wanted.items():
            [(_, frame)] = video.sample(path, 1, index, index)
            cv2.imwrite(str(CACHE / f"{name}.{key}.png"), resize(crop(frame, box), FRAME_SIDE))
        (CACHE / f"{name}.json").write_text(json.dumps({"box": box, "frames": wanted}))
        print(name, box, flush=True)


def probe(vlm: VLM) -> None:
    """Both questions of each probe on each video, as prompt v9 asks them; the answer is kept
    only when both agree. Prints the tally of (check / problem -> answer), "yes" = good."""
    prompts = base.load(STROKE, "prompts/v9.yaml")["checks"]
    for probe_name, (frame, check, problem) in PROBES.items():
        tally = collections.defaultdict(collections.Counter)
        for meta in sorted(CACHE.glob("*.json")):
            name = meta.name.removesuffix(".json")
            image_path = CACHE / f"{name}.{frame}.png"
            if not image_path.exists():
                continue
            image, votes = cv2.imread(str(image_path)), []
            for question, flip in ((check, False), (problem, True)):
                user = Template(prompts["user"]).substitute(question=question)
                answer = vlm.generate_json(prompts["system"], user, [image], base.check_schema(),
                                           **GENERATION).data["answer"]  # fmt: skip
                votes.append({"yes": "no", "no": "yes"}.get(answer, answer) if flip else answer)
            final = votes[0] if votes[0] == votes[1] else "unclear"
            tally["pro" if name.startswith("pro") else "jc"][f"{votes[0]}/{votes[1]}->{final}"] += 1
        print(
            probe_name, {group: dict(sorted(t.items())) for group, t in tally.items()}, flush=True
        )


if __name__ == "__main__":
    model = VLM(CONFIG["llm"]["model"])
    cache(model) if sys.argv[1] == "cache" else probe(model)
