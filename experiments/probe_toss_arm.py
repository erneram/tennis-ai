"""Can the model see the tossing arm drop? toss_arm_drops_early on the leg-drive frames of the
v10 runs, as v10 asks it and with two changes, without running the whole system. Source of the
probe numbers in the result of strokes/serve/prompts/v10.yaml.

    uv run python -m experiments.probe_toss_arm

A = the frame as v10 saw it (cropped to the player's reach); B = the upper body only, from a
box the model draws on that frame, enlarged to the same size (strokes/base.py, upper_body).
Q1 = v10's two yes/no questions; Q2 = one neutral question: where is the tossing hand. Frames
saved in data/processed/probe/v10/.

    uv run python -m experiments.probe_toss_arm offsets [model] [version] [seconds...]

offsets (v13): the frames a fixed time before the contact frame of the v12 runs, cropped as
the system crops them, asked the neutral question of `version` (default v12) on the upper
body. Counts per offset,
group and half (dev: JC s8-s17 and the first 6 professional serves; test: the rest). Frames
saved in data/processed/probe/offsets/<model>-<version>/. Results in the headers of
strokes/serve/prompts/v13.yaml (why 0.1 s), v14.yaml and v15.yaml. With
mlx-community/Qwen3-VL-8B-Instruct-4bit and v12's question, no frame from 0.4 to 0.05 s got
"at shoulder height or below" (JC unclear on 8 of 21 at 0.1 s): the larger model does not see
the drop. One video frame either side of 0.1 s (offsets ... v12 0.117 0.083), the answer
changes: at 0.117 s every serve "above the head" (1 unclear); at 0.083 s JC "at shoulder height
or below" on 8 of 21, but also pro03_s04. A vote of the three frames finds 3 JC serves: the
detection depends on one frame.
"""

import collections
import json
import sys
from string import Template

import cv2

from core import ROOT, load_config, video
from core.analysis import FRAME_SIDE, crop, resize
from core.llm import VLM
from strokes import base

OUT = ROOT / "data" / "processed" / "probe" / "v10"  # crops of players' videos: not in Git
RUNS = ["20261101-231243-full", "20261101-231849-full"]
V12_RUNS = ["20261102-002057-full", "20261102-002636-full"]
OFFSETS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)  # seconds before contact
DEV = {f"jc-s{n}" for n in range(8, 18)} | {"pro01_s01", "pro01_s02", "pro02_s01", "pro03_s01",
                                            "pro03_s02", "pro03_s03"}  # fmt: skip
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
            upper, _ = base.upper_body(frame, vlm, PROMPTS["player"], **GENERATION)
            images = {"A": resize(crop(frame, record.get("box")), FRAME_SIDE), "B": upper}
            row = {"video": path.stem, "before_contact_s": round((contact - index) / fps, 2)}
            for variant, image in images.items():
                cv2.imwrite(str(OUT / f"{path.stem}.{variant}.jpg"), image)
                row[variant] = ask(vlm, image)
                for question in ("Q1", "Q2"):
                    tally[f"{variant} {question} {group}"][row[variant][question]] += 1
            print(json.dumps(row), flush=True)
    for key, counts in sorted(tally.items()):
        print(key, dict(counts))


def offsets(model: str, version: str = "v12", *only: str) -> None:
    vlm, prompts = VLM(model), base.load(base.get("serve"), f"prompts/{version}.yaml")
    spec, ask = prompts["checks"]["questions"]["toss_arm_drops_early"], prompts["checks"]["options"]
    options = [*spec["options"], "unclear"]
    user = Template(ask["user"]).substitute(question=spec["question"], options="; ".join(options))
    out = ROOT / "data" / "processed" / "probe" / "offsets" / f"{model.split('/')[-1]}-{version}"
    out.mkdir(parents=True, exist_ok=True)
    tally = collections.defaultdict(collections.Counter)
    for run in V12_RUNS:
        for line in (ROOT / "eval" / "serve" / "runs" / f"{run}.jsonl").open():
            record = json.loads(line)
            if not record.get("key_frames"):
                continue
            path, contact = ROOT / record["video"], record["key_frames"]["contact"]
            fps = video.probe(path).fps
            wanted = {max(0, contact - round(s * fps)): s for s in map(float, only or OFFSETS)}
            frames = {i: f for i, f in video.iter_frames(path) if i in wanted}
            group = "pro" if path.stem.startswith("pro") else "jc"
            half = "dev" if path.stem in DEV else "test"
            row = {"video": path.stem}
            for index, seconds in sorted(wanted.items(), key=lambda item: -item[1]):
                image = resize(crop(frames[index], record.get("box")), FRAME_SIDE)
                upper, _ = base.upper_body(image, vlm, prompts["player"], **GENERATION)
                cv2.imwrite(str(out / f"{path.stem}.{seconds}.jpg"), upper)
                hand = vlm.generate_json(ask["system"], user, [upper],
                                         base.check_schema(tuple(options)), **GENERATION).data  # fmt: skip
                answer = spec["options"].get(hand["answer"], "unclear")
                row[str(seconds)] = [answer, hand["seen"]]
                tally[f"-{seconds}s {group} {half}"][answer] += 1
            print(json.dumps(row), flush=True)
    for key, counts in sorted(tally.items()):
        print(key, dict(counts))


if __name__ == "__main__":
    if sys.argv[1:2] == ["offsets"]:
        offsets(*(sys.argv[2:] or [CONFIG["llm"]["model"]]))
    else:
        main()
