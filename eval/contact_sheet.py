"""One image per video with 20 frames around the serve, cropped to the moving player, for a
review by eye. Input of eval/serve/claude_labels.csv. No language model: the serve is found as
the peak of motion (the racket swing at contact), and the crop as the region that moves.

    uv run python -m eval.contact_sheet data/raw/jc-s8.mp4 ...   # -> data/processed/sheets/
"""

import sys
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from core import ROOT, video
from core.analysis import label

OUT = ROOT / "data" / "processed" / "sheets"


def sheet(path: Path, cols: int = 5, rows: int = 4, tile: int = 300) -> Path:
    info = video.probe(path)
    scale = 320 / max(info.width, info.height)

    def gray(frame):
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    grays = {i: gray(frame) for i, frame in video.iter_frames(path)}
    keys = sorted(grays)
    moved = {b: cv2.absdiff(grays[b], grays[a]) > 25 for a, b in pairwise(keys)}
    energy = np.convolve([moved[i].sum() for i in keys[1:]], np.ones(3) / 3, "same")
    peak = keys[1:][int(energy.argmax())]
    first = max(0, peak - round(1.4 * info.fps))
    last = min(keys[-1], peak + round(0.7 * info.fps))
    total = sum(moved[i].astype(np.uint16) for i in keys[1:] if first <= i <= last)
    ys, xs = np.nonzero(total > 1)
    x0, x1 = np.percentile(xs, [1, 99]) / scale
    y0, y1 = np.percentile(ys, [1, 99]) / scale
    side, cx, cy = max(x1 - x0, y1 - y0) * 1.15, (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(info.width, cx + side / 2)), int(min(info.height, cy + side / 2))

    tiles = []
    for i, frame in video.sample(path, cols * rows, first, last):
        part = frame[y0:y1, x0:x1]
        s = tile / max(part.shape[:2])
        part = cv2.resize(part, (round(part.shape[1] * s), round(part.shape[0] * s)),
                          interpolation=cv2.INTER_AREA)  # fmt: skip
        canvas = np.zeros((tile, tile, 3), np.uint8)
        canvas[: part.shape[0], : part.shape[1]] = part
        tiles.append(label(canvas, f"{i} ({(i - peak) / info.fps:+.2f}s)", 0.06))
    grid = cv2.vconcat([cv2.hconcat(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
    out = OUT / f"{path.stem}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(sheet(Path(arg)))
