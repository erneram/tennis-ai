"""Extract MediaPipe landmarks from serve videos.

    uv run python -m core.pose data/raw/*.mp4 [--model heavy] [--sheet] [--out DIR]

Writes data/processed/pose/<video>.npz (or DIR/<video>.npz) (see core.pose.extract) and, with --sheet, <video>.jpg:
10 frames around contact with the skeleton drawn, to check the landmarks by eye.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from core import ROOT, load_config, pose, video
from core.analysis import label

OUT = ROOT / "data" / "processed" / "pose"


def sheet(path: Path, result: dict, out: Path) -> None:
    x0, y0, x1, y1 = result["box"]
    tiles = []
    for i, frame in video.sample(path, 10, result["first"], result["last"]):
        part = pose.draw(frame, result["keypoints"][i])[y0:y1, x0:x1]
        part = cv2.resize(part, (300, round(300 * part.shape[0] / part.shape[1])))
        text = f"{i} ({(i - result['contact']) / result['fps']:+.2f}s)"
        tiles.append(label(part, text, 0.06))
    cv2.imwrite(str(out), cv2.vconcat([cv2.hconcat(tiles[:5]), cv2.hconcat(tiles[5:])]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--model", default=load_config()["pose"]["model"])
    parser.add_argument("--sheet", action="store_true", help="also write a check image")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.videos:
        start = time.perf_counter()
        result = pose.extract(path, args.model)
        np.savez_compressed(args.out / f"{path.stem}.npz", **result)
        if args.sheet:
            sheet(path, result, args.out / f"{path.stem}.jpg")
        found = ~np.isnan(result["keypoints"][:, 0, 0])
        serve = found[result["first"] : result["last"] + 1]
        print(f"{path.name}: {len(found)} frames in {time.perf_counter() - start:.0f}s, person "
              f"in {found.mean():.0%} of all frames, {serve.mean():.0%} around contact")  # fmt: skip


if __name__ == "__main__":
    main()
