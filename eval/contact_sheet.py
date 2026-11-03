"""One image per video with 20 frames around the serve, cropped to the moving player, for a
review by eye. Input of eval/serve/claude_labels.csv. No language model: the serve is found as
the peak of motion (the racket swing at contact), and the crop as the region that moves.

    uv run python -m eval.contact_sheet data/raw/jc-s8.mp4 ...   # -> data/processed/sheets/
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from core import ROOT, video
from core.analysis import label

OUT = ROOT / "data" / "processed" / "sheets"


def sheet(path: Path, cols: int = 5, rows: int = 4, tile: int = 300) -> Path:
    info = video.probe(path)
    m = video.motion(path, before=1.4, after=0.7)
    x0, y0, x1, y1 = m.box
    tiles = []
    for i, frame in video.sample(path, cols * rows, m.first, m.last):
        part = frame[y0:y1, x0:x1]
        s = tile / max(part.shape[:2])
        part = cv2.resize(part, (round(part.shape[1] * s), round(part.shape[0] * s)),
                          interpolation=cv2.INTER_AREA)  # fmt: skip
        canvas = np.zeros((tile, tile, 3), np.uint8)
        canvas[: part.shape[0], : part.shape[1]] = part
        tiles.append(label(canvas, f"{i} ({(i - m.peak) / info.fps:+.2f}s)", 0.06))
    grid = cv2.vconcat([cv2.hconcat(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
    out = OUT / f"{path.stem}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(sheet(Path(arg)))
