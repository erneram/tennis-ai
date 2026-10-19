"""Video loading and frame extraction."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
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
