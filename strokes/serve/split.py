"""Cut a long practice session into one clip per serve, for labeling.

    uv run python -m strokes.serve.split find data/raw/uncut/*.MOV
    uv run python -m strokes.serve.split cut data/raw/uncut/ea_2.MOV --name ea-2 --drop 1,5,8

find: MediaPipe Pose (full) follows the player, RATE frames per second. Every stretch with a wrist
dataset.TOSS torso lengths above the shoulders for MIN_UP seconds or more is a candidate, at its
highest wrist (the top of the toss); candidates closer than APART seconds are one. Writes
data/processed/split/<video>.json and <video>.jpg: each candidate 0.8 s before, at, and 0.4 and
0.8 s after that moment, to check by eye before cutting. The check is needed: on an empty court
MediaPipe can see a person in a bag, and a toss the player catches without swinging is a
candidate too. The tracker starts again when the person it follows has not moved for STILL
seconds, so it does not stay on such a bag while the player serves.

cut: data/raw/<name>-sNN.mp4 for every candidate not in --drop, plus --add (seconds of serves the
check found missing), from BEFORE seconds before the top of the toss to AFTER seconds after it:
H.264 with the phone's rotation applied (ffmpeg). Appends clip, source and times to cuts.csv next
to the source video. Never overwrites a clip.
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from core import ROOT, pose, video
from core.analysis import label
from strokes.serve import dataset

OUT = ROOT / "data" / "processed" / "split"
RAW = ROOT / "data" / "raw"
RATE = 15  # samples per second
MIN_UP, GAP, APART = 0.4, 0.3, 3.0  # seconds: shortest toss, longest break in one, between serves
STILL = 2.0  # seconds without moving after which the tracker starts again
BEFORE, AFTER = 3.0, 2.0  # seconds of the clip around the top of the toss
MOMENTS = (-0.8, 0.0, 0.4, 0.8)  # seconds from the top of the toss shown in the check image
JOINT = {name: pose.LANDMARKS.index(name) for name in pose.LANDMARKS}


def follow(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """fps, the sampled frame numbers and (samples, 33, 2) landmarks in pixels (NaN: no one)."""
    info = video.probe(path)
    step = max(1, round(info.fps / RATE))
    still = round(STILL * RATE)
    frames, points, landmarker, followed = [], [], pose.load("full"), 0
    try:
        for i, frame in video.iter_frames(path, step):
            if followed >= still:
                hips = np.array(points[-still:])[:, [JOINT["left_hip"], JOINT["right_hip"]]]
                if not np.isnan(hips).any() and hips.std(0).max() < 2:
                    landmarker.close()  # a person still for STILL s is not the player: a bag
                    landmarker, followed = pose.load("full"), 0
            followed += 1
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            found = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(i * 1000 / info.fps)
            ).pose_landmarks
            height, width = frame.shape[:2]
            frames.append(i)
            points.append([(p.x * width, p.y * height) for p in found[0]] if found
                          else np.full((len(pose.LANDMARKS), 2), np.nan))  # fmt: skip
    finally:
        landmarker.close()
    return info.fps, np.array(frames), np.array(points, dtype=float)


def candidates(fps: float, frames: np.ndarray, points: np.ndarray) -> list[dict]:
    """One candidate per toss: seconds and frame of the highest wrist, seconds up and height."""
    shoulders = points[:, [JOINT["left_shoulder"], JOINT["right_shoulder"]]].mean(1)
    hips = points[:, [JOINT["left_hip"], JOINT["right_hip"]]].mean(1)
    torso = np.linalg.norm(shoulders - hips, axis=1)
    near = 2 * RATE  # torso over +-2 s: the player walks closer and farther
    torso = np.array([np.nan if np.isnan(torso[max(0, k - near) : k + near]).all()
                      else np.nanmedian(torso[max(0, k - near) : k + near])
                      for k in range(len(torso))])  # fmt: skip
    wrists = points[:, [JOINT["left_wrist"], JOINT["right_wrist"]], 1]
    above = (shoulders[:, 1, None] - wrists) / torso[:, None]  # image y grows downward
    height = np.nan_to_num(np.fmax(above[:, 0], above[:, 1]), nan=-np.inf)
    up = np.flatnonzero(height >= dataset.TOSS)
    found = []
    for run in np.split(up, np.flatnonzero(np.diff(up) > GAP * RATE) + 1) if len(up) else []:
        seconds_up = (run[-1] - run[0] + 1) / RATE
        if seconds_up < MIN_UP:
            continue
        top = run[0] + int(height[run[0] : run[-1] + 1].argmax())
        candidate = {"seconds": round(frames[top] / fps, 3), "frame": int(frames[top]),
                     "up": round(seconds_up, 2), "height": round(float(height[top]), 2)}  # fmt: skip
        if found and candidate["seconds"] - found[-1]["seconds"] < APART:
            if candidate["up"] > found[-1]["up"]:  # one serve: keep the longest stretch up
                found[-1] = candidate
        else:
            found.append(candidate)
    return found


def check_image(path: Path, fps: float, frames: np.ndarray, points: np.ndarray,
                found: list[dict], out: Path) -> None:  # fmt: skip
    """Each candidate in a row: MOMENTS around the top of the toss, cropped to the player."""
    wanted: dict[int, list[tuple[int, int]]] = {}
    for n, c in enumerate(found):
        for m, offset in enumerate(MOMENTS):
            wanted.setdefault(max(0, c["frame"] + round(offset * fps)), []).append((n, m))
    tiles = {}
    for i, frame in video.iter_frames(path):
        if i > max(wanted, default=-1):
            break
        for n, m in wanted.get(i, []):
            seen = points[int(np.abs(frames - found[n]["frame"]).argmin())]
            if np.isnan(seen).all():
                part = frame
            else:
                (x0, y0), (x1, y1) = np.nanmin(seen, 0), np.nanmax(seen, 0)
                side = max(x1 - x0, y1 - y0) * 1.6
                left, top = int(max(0, (x0 + x1 - side) / 2)), int(max(0, (y0 + y1 - side) / 2))
                part = frame[top : top + int(side), left : left + int(side)]
            text = f"{n + 1}: {found[n]['seconds']:.1f} s" if m == 0 else f"{MOMENTS[m]:+.1f} s"
            tiles[n, m] = label(cv2.resize(part, (200, 200)), text, 0.08)
    if found:
        rows = [cv2.hconcat([tiles[n, m] for m in range(len(MOMENTS))]) for n in range(len(found))]
        rows += [np.zeros_like(rows[0])] * (len(rows) % 2)
        cv2.imwrite(str(out), cv2.vconcat([cv2.hconcat(rows[r : r + 2])
                                           for r in range(0, len(rows), 2)]))  # fmt: skip


def find(paths: list[Path]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in paths:
        fps, frames, points = follow(path)
        found = candidates(fps, frames, points)
        (OUT / f"{path.stem}.json").write_text(json.dumps({"fps": fps, "candidates": found}))
        check_image(path, fps, frames, points, found, OUT / f"{path.stem}.jpg")
        print(f"{path.name}: {len(found)} candidates, check {OUT / path.stem}.jpg", flush=True)


def cut(path: Path, name: str, drop: set[int], add: list[float]) -> None:
    found = json.loads((OUT / f"{path.stem}.json").read_text())["candidates"]
    tops = sorted([c["seconds"] for n, c in enumerate(found, 1) if n not in drop] + add)
    clips = [RAW / f"{name}-s{k:02d}.mp4" for k in range(1, len(tops) + 1)]
    if taken := [c.name for c in clips if c.exists()]:
        raise SystemExit(f"already in data/raw: {', '.join(taken)}")
    log = path.parent / "cuts.csv"
    if not log.exists():
        log.write_text("clip,source,top_s,start_s,end_s\n")
    for clip, top in zip(clips, tops):
        start = max(0.0, top - BEFORE)
        subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path),
                        "-t", f"{top + AFTER - start:.3f}", "-map", "0:v:0", "-map", "0:a:0?",
                        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-movflags", "+faststart", str(clip)], check=True)  # fmt: skip
        with log.open("a") as file:
            file.write(f"{clip.name},{path.name},{top:.3f},{start:.3f},{top + AFTER:.3f}\n")
        print(f"{clip.name}: {path.name} at {top:.2f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("find").add_argument("videos", type=Path, nargs="+")
    one = steps.add_parser("cut")
    one.add_argument("video", type=Path)
    one.add_argument("--name", required=True, help="clip prefix, player first: ea-2, jc-post1")
    one.add_argument("--drop", default="", help="candidate numbers that are not serves: 1,5,8")
    one.add_argument("--add", default="", help="seconds of serves find missed: 262.0")
    args = parser.parse_args()
    if args.step == "find":
        find(args.videos)
    else:
        numbers = lambda text, kind: [kind(v) for v in text.split(",") if v]
        cut(args.video, args.name, set(numbers(args.drop, int)), numbers(args.add, float))


if __name__ == "__main__":
    main()
