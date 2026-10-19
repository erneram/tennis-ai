"""Run pose estimation on a video and write the keypoint overlay.

    uv run python -m core.pose data/raw/serve.mp4 [--backend yolo] [--step 2]

Writes <out>/<video>_<backend>.mp4 (overlay) and <out>/<video>_<backend>.npz
(keypoints, frame_indices, fps, backend).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from core import pose, video

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())["pose"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--backend", default=config["backend"], choices=["mediapipe", "yolo"])
    parser.add_argument("--step", type=int, default=1, help="process every n-th frame")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    if args.step < 1:
        parser.error("--step must be >= 1")

    info = video.probe(args.video)
    detect = pose.load(args.backend, config[args.backend])
    indices, keypoints = [], []

    def overlay():
        for index, frame in video.iter_frames(args.video, args.step):
            frame_keypoints = detect(frame, int(index * 1000 / info.fps))
            indices.append(index)
            keypoints.append(frame_keypoints)
            yield pose.draw(frame, frame_keypoints)

    name = f"{args.video.stem}_{args.backend}"
    start = time.perf_counter()
    count = video.write_video(args.out / f"{name}.mp4", overlay(), info.fps / args.step)
    elapsed = time.perf_counter() - start
    if not count:
        raise SystemExit(f"No frames could be decoded from {args.video}")

    stacked = np.stack(keypoints)
    np.savez(
        args.out / f"{name}.npz",
        keypoints=stacked,
        frame_indices=np.array(indices),
        fps=info.fps,
        backend=args.backend,
    )
    detected = np.mean(stacked[:, :, 2].max(axis=1) > 0)
    print(
        f"{count} frames in {elapsed:.1f}s ({count / elapsed:.1f} fps), "
        f"person in {detected:.0%} of frames -> {args.out / name}.mp4"
    )


if __name__ == "__main__":
    main()
