"""Training data for the serve model of the Final Project: MediaPipe landmarks around contact.

    uv run python -m strokes.serve.dataset [--labels eval/serve/labels.csv]

Reads data/processed/pose/<video>.npz (written by `python -m core.pose`) and writes
data/processed/serve_dataset.npz with one window per video: 64 frames at 30 fps aligned on
contact, centered on the hips at contact and in torso lengths, with a mask of the points
MediaPipe did not see. Videos without a valid label stay in the file for pretraining.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from core import ROOT, pose

POSE_DIR = ROOT / "data" / "processed" / "pose"
OUT = ROOT / "data" / "processed" / "serve_dataset.npz"
LABELS = ROOT / "eval" / "serve" / "labels.csv"

FPS = 30
BEFORE, AFTER = 48, 15  # frames before and after contact: from the stance to the landing
JOINTS = (
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)  # fmt: skip
SHOULDERS, HIPS = [1, 2], [7, 8]  # positions in JOINTS
# Never used to train or pretrain: the session filmed on Nov 1 and one professional
TEST_VIDEOS = {f"jc-s{i}" for i in range(28, 44)}
TEST_PLAYERS = {"pro01"}


def contact(keypoints: np.ndarray, fps: float) -> int:
    """Frame of contact: the highest wrist in the second after the toss.

    The toss is the longest stretch with a wrist 0.4 torso lengths above the shoulders: the tossing
    arm stays up while the ball rises. The frame with the most motion (core.video.motion) is wrong
    when someone walks by or the player runs in after the serve."""
    points = keypoints[:, [pose.LANDMARKS.index(j) for j in JOINTS]]
    shoulders, hips = points[:, SHOULDERS, :2].mean(1), points[:, HIPS, :2].mean(1)
    torso = np.nanmedian(np.linalg.norm(shoulders - hips, axis=1))
    above = (shoulders[:, None, 1] - points[:, [5, 6], 1]) / torso  # image y grows downward
    height = np.nan_to_num(np.fmax(above[:, 0], above[:, 1]), nan=-np.inf)
    up = np.flatnonzero(height >= 0.4)
    if not len(up):
        raise ValueError("no wrist above the shoulders")
    toss_end = max(np.split(up, np.flatnonzero(np.diff(up) > 1) + 1), key=len)[-1]
    after = height[toss_end + 1 : toss_end + 1 + round(fps)]
    if not len(after):
        raise ValueError("the video ends during the toss")
    return int(toss_end + 1 + after.argmax())


def window(keypoints: np.ndarray, fps: float, contact: int) -> tuple[np.ndarray, np.ndarray]:
    """(BEFORE + 1 + AFTER, len(JOINTS), 2) x, y at FPS around contact, with the hips at contact at
    the origin and the median torso as unit, and a mask of the points seen (visibility 0.5 or more,
    inside the video). Unseen points are interpolated in time from the frames where they were seen.
    """
    points = keypoints[:, [pose.LANDMARKS.index(j) for j in JOINTS]]
    seen = points[..., 3] >= 0.5  # NaN compares False
    times = np.arange(len(points)) / fps
    wanted = contact / fps + np.arange(-BEFORE, AFTER + 1) / FPS
    x = np.zeros((len(wanted), len(JOINTS), 2), np.float32)
    for j in range(len(JOINTS)):
        if seen[:, j].any():
            for c in range(2):
                x[:, j, c] = np.interp(wanted, times[seen[:, j]], points[seen[:, j], j, c])
    nearest = np.clip(np.round(wanted * fps).astype(int), 0, len(points) - 1)
    inside = (wanted >= 0) & (wanted <= times[-1])
    mask = seen[nearest] & inside[:, None]
    hips = x[:, HIPS].mean(1)
    torso = np.median(np.linalg.norm(x[:, SHOULDERS].mean(1) - hips, axis=1))
    return (x - hips[BEFORE]) / torso, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=LABELS)
    args = parser.parse_args()
    rows = csv.DictReader(args.labels.read_text().splitlines())
    labels = {Path(r["video"]).stem: r for r in rows}
    columns = ("x", "mask", "video", "player", "valid", "category", "split", "contact", "video_fps")
    data = {name: [] for name in columns}
    for path in sorted(POSE_DIR.glob("*.npz")):
        result = np.load(path)
        keypoints, fps = result["keypoints"], float(result["fps"])
        try:
            frame = contact(keypoints, fps)
        except ValueError as error:
            print(f"{path.stem}: skipped, {error}")
            continue
        if abs(frame - int(result["contact"])) > 0.3 * fps:
            print(f"{path.stem}: contact at frame {frame}, the most motion at {result['contact']}")
        x, mask = window(keypoints, fps, frame)
        label = labels.get(path.stem)
        player = re.split(r"[-_]", path.stem)[0]
        valid = int(label["valid"]) if label and label["valid"] else -1  # -1: not labeled
        category = label["category"] if valid == 1 else ""
        test = path.stem in TEST_VIDEOS or player in TEST_PLAYERS
        split = "test" if test else "train"
        values = (x, mask, path.stem, player, valid, category, split, frame, fps)
        for name, value in zip(columns, values):
            data[name].append(value)
    data = {name: np.array(values) for name, values in data.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **data, joints=np.array(JOINTS), fps=FPS, before=BEFORE)
    seen = data["mask"].mean()
    print(f"{len(data['video'])} videos, {seen:.0%} of points seen; labels {args.labels.name}")
    for split in ("train", "test"):
        chosen = (data["split"] == split) & (data["valid"] == 1)
        names, counts = np.unique(data["category"][chosen], return_counts=True)
        labeled = dict(zip(names.tolist(), counts.tolist()))
        print(f"  {split}: {(data['split'] == split).sum()} videos, labeled {labeled}")


if __name__ == "__main__":
    main()
