"""Compare pose backends on the same videos: speed, detection rate, confidence, and jitter.

    uv run python -m experiments.compare_pose data/raw/*.mp4    # needs: uv sync --group yolo

Jitter is the mean frame-to-frame acceleration of the core joints, in torso lengths: on the
same video, a lower value means a steadier skeleton. Results are appended to
experiments/pose_comparison.csv.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from core import ROOT, load_config, metrics, pose, video
from core.analysis import CORE_JOINTS

OUT = ROOT / "experiments" / "pose_comparison.csv"
FIELDS = ["date", "video", "backend", "model", "frames", "fps", "detected", "confidence", "jitter"]


def run(path: Path, backend: str, model: str) -> dict:
    info = video.probe(path)
    detect = pose.load(backend, model)  # loaded (and warmed up) before timing
    keypoints = []
    start = time.perf_counter()
    for index, frame in video.iter_frames(path):
        keypoints.append(detect(frame, int(index * 1000 / info.fps)))
    elapsed = time.perf_counter() - start

    keypoints = np.stack(keypoints)
    core = keypoints[:, CORE_JOINTS]
    xy = core[..., :2].copy()
    xy[core[..., 2] < 0.5] = np.nan
    torso = metrics.torso_length(metrics.clean(keypoints, info.fps))
    acceleration = np.linalg.norm(np.diff(xy, n=2, axis=0), axis=-1) / torso
    return {
        "date": f"{datetime.now():%Y-%m-%d}",
        "video": path.name,
        "backend": backend,
        "model": model,
        "frames": len(keypoints),
        "fps": round(len(keypoints) / elapsed, 1),
        "detected": round(float((keypoints[:, :, 2].max(axis=1) > 0).mean()), 3),
        "confidence": round(float(core[..., 2].mean()), 3),
        "jitter": round(float(np.nanmean(acceleration)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--backends", nargs="+", default=["mediapipe", "yolo"])
    args = parser.parse_args()

    config = load_config()["pose"]
    rows = [run(path, b, config[b]) for path in args.videos for b in args.backends]
    new_file = not OUT.exists()
    with open(OUT, "a", newline="") as file:
        writer = csv.DictWriter(file, FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

    for backend in args.backends:
        mine = [r for r in rows if r["backend"] == backend]
        means = {k: round(float(np.mean([r[k] for r in mine])), 3)
                 for k in ("fps", "detected", "confidence", "jitter")}  # fmt: skip
        print(backend, means)
    print(f"{len(rows)} rows appended to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
