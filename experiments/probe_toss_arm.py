"""Can the model see the tossing arm drop? toss_arm_drops_early on the leg-drive frames of the
v10 runs, as v10 asks it and with two changes, without running the whole system. Source of the
probe numbers in the result of strokes/serve/prompts/v10.yaml.

    uv run python -m experiments.probe_toss_arm

A = the frame as v10 saw it (cropped to the player's reach); B = the upper body only, from a
box the model draws on that frame, enlarged to the same size. Q1 = v10's two yes/no questions;
Q2 = one neutral question: where is the tossing hand. Frames saved in data/processed/probe/v10/.
"""

import collections
import json
from string import Template

import cv2

from core import ROOT, load_config, video
from core.analysis import FRAME_SIDE, crop, resize
from core.llm import VLM
from strokes import base

OUT = ROOT / "data" / "processed" / "probe" / "v10"  # crops of players' videos: not in Git
RUNS = ["20261101-231243-full", "20261101-231849-full"]
CONFIG = load_config()
GENERATION = {k: CONFIG["llm"][k] for k in ("temperature", "top_p", "max_tokens", "seed")}
PROMPTS = base.load(base.get("serve"), "prompts/v10.yaml")
HAND = ("above the head", "between the shoulder and the top of the head",
        "at shoulder height or below", "unclear")  # fmt: skip
HAND_SYSTEM = (
    "You look at ONE frame of a tennis serve filmed from the side, and answer ONE question about it.\n"
    'First write "seen": one short sentence, in English, describing only what the frame shows\n'
    "about the body part in the question.\n"
    'Then "answer": the option that matches the frame, or "unclear" if you cannot tell: the player\n'
    "is small, blurry or hidden."
)
HAND_USER = ("Question: Where is the tossing hand (the hand that released the ball, not the hand "
             f"holding the racket)?\nOptions: {'; '.join(HAND)}\n\nReply with the JSON object only.")  # fmt: skip


def upper_body(vlm: VLM, frame):
    """The player's box on this frame (v10's player prompt), cut to the head, shoulders and
    arms, and enlarged to FRAME_SIDE."""
    player = PROMPTS["player"]
    data = vlm.generate_json(player["system"], player["user"], [frame], base.player_schema(),
                             **GENERATION).data  # fmt: skip
    x0, y0, x1, y1 = (v / 1000 for v in data["bbox_2d"])
    w, h = x1 - x0, y1 - y0
    part = crop(frame, (max(0, x0 - 0.15 * w), max(0, y0 - 0.15 * h), min(1, x1 + 0.15 * w),
                        min(1, y0 + 0.6 * h)))  # fmt: skip
    scale = FRAME_SIDE / max(part.shape[:2])
    return cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def ask(vlm: VLM, image) -> dict:
    checks = PROMPTS["checks"]
    questions, votes = checks["questions"]["toss_arm_drops_early"], []
    for question, flip in ((questions["check"], False), (questions["problem"], True)):
        user = Template(checks["user"]).substitute(question=question)
        answer = vlm.generate_json(checks["system"], user, [image], base.check_schema(),
                                   **GENERATION).data["answer"]  # fmt: skip
        votes.append({"yes": "no", "no": "yes"}.get(answer, answer) if flip else answer)
    hand = vlm.generate_json(HAND_SYSTEM, HAND_USER, [image], base.check_schema(HAND),
                             **GENERATION).data  # fmt: skip
    return {"Q1": "/".join(votes), "Q2": hand["answer"], "seen": hand["seen"]}


def main() -> None:
    vlm = VLM(CONFIG["llm"]["model"])
    OUT.mkdir(parents=True, exist_ok=True)
    tally = collections.defaultdict(collections.Counter)
    for run in RUNS:
        for line in (ROOT / "eval" / "serve" / "runs" / f"{run}.jsonl").open():
            record = json.loads(line)
            if not record.get("key_frames"):
                continue
            path, index = ROOT / record["video"], record["key_frames"]["leg_drive"]
            [(_, frame)] = video.sample(path, 1, index, index)
            fps, contact = video.probe(path).fps, record["key_frames"]["contact"]
            group = "pro" if path.stem.startswith("pro") else "jc"
            images = {"A": resize(crop(frame, record.get("box")), FRAME_SIDE),
                      "B": upper_body(vlm, frame)}  # fmt: skip
            row = {"video": path.stem, "before_contact_s": round((contact - index) / fps, 2)}
            for variant, image in images.items():
                cv2.imwrite(str(OUT / f"{path.stem}.{variant}.jpg"), image)
                row[variant] = ask(vlm, image)
                for question in ("Q1", "Q2"):
                    tally[f"{variant} {question} {group}"][row[variant][question]] += 1
            print(json.dumps(row), flush=True)
    for key, counts in sorted(tally.items()):
        print(key, dict(counts))


if __name__ == "__main__":
    main()
