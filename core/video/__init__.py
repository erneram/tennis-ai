"""Video loading and frame extraction."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int


def _open(path: str | Path) -> cv2.VideoCapture:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    return cap


def probe(path: str | Path) -> VideoInfo:
    """Read video metadata. Raises ValueError if the file is not a usable video."""
    cap = _open(path)
    try:
        info = VideoInfo(
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()
    if info.fps <= 0 or info.frame_count <= 0:
        raise ValueError(f"Video has no frames or unknown fps: {path}")
    return info


def iter_frames(path: str | Path, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) for every `step`-th frame."""
    cap = _open(path)
    try:
        index = 0
        while cap.grab():
            if index % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield index, frame
            index += 1
    finally:
        cap.release()


def sample(path: str | Path, count: int, first: int = 0,
           last: int | None = None) -> list[tuple[int, np.ndarray]]:  # fmt: skip
    """`count` frames spread evenly from frame `first` to frame `last` (default: the end of the
    video), as (frame_index, BGR frame).

    Frames are counted by decoding them: phone videos often report more frames than they have.
    """
    if last is None:
        cap = _open(path)
        try:
            last = -1
            while cap.grab():
                last += 1
        finally:
            cap.release()
    size = last - first + 1
    wanted = set(np.linspace(first, last, min(count, size)).round().astype(int).tolist())
    frames = []
    for index, frame in iter_frames(path):
        if index > last:
            break
        if index in wanted:
            frames.append((index, frame))
    return frames


@dataclass(frozen=True)
class Motion:
    peak: int  # frame with the most motion: the racket swing at contact
    first: int
    last: int
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels: a square around what moves


def motion(path: str | Path, before: float, after: float) -> Motion:
    """Find the peak of motion and a square box around the region that moves from `before`
    seconds before it to `after` seconds after it. No model: frame differences at 320 px."""
    info = probe(path)
    scale = 320 / max(info.width, info.height)

    def gray(frame):
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    grays = {i: gray(frame) for i, frame in iter_frames(path)}
    keys = sorted(grays)
    moved = {b: cv2.absdiff(grays[b], grays[a]) > 25 for a, b in pairwise(keys)}
    energy = np.convolve([moved[i].sum() for i in keys[1:]], np.ones(3) / 3, "same")
    peak = keys[1:][int(energy.argmax())]
    first = max(0, peak - round(before * info.fps))
    last = min(keys[-1], peak + round(after * info.fps))
    total = sum(moved[i].astype(np.uint16) for i in keys[1:] if first <= i <= last)
    ys, xs = np.nonzero(total > 1)
    x0, x1 = np.percentile(xs, [1, 99]) / scale
    y0, y1 = np.percentile(ys, [1, 99]) / scale
    side, cx, cy = max(x1 - x0, y1 - y0) * 1.15, (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(info.width, cx + side / 2)), int(min(info.height, cy + side / 2))
    return Motion(peak, first, last, (x0, y0, x1, y1))


def write_video(path: str | Path, frames: Iterable[np.ndarray], fps: float) -> int:
    """Write BGR frames to an H.264 .mp4 file and return the number of frames written."""
    writer = None
    count = 0
    try:
        for frame in frames:
            if writer is None:
                # H.264 needs even sizes
                height, width = (side - side % 2 for side in frame.shape[:2])
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264: plays in browsers
                writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            writer.write(np.ascontiguousarray(frame[:height, :width]))
            count += 1
    finally:
        if writer is not None:
            writer.release()
    return count
