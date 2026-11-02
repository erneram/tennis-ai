"""Video side of one analysis: quality checks and evenly spaced frames for the language model.

There is no pose model. The language model finds the phases in these frames and judges them
(strokes/base.py).
"""

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from core import video

# Quality thresholds. ponytail: initial guesses, tune them on the first recorded videos.
MIN_FPS = 24
MIN_SECONDS = 1.0
FRAME_SIDE = 1024  # longest side of the kept frames: the player stays large enough to judge


class InsufficientEvidence(Exception):
    """The video does not show enough to give feedback; the player must record again."""


@dataclass
class Frames:
    video: str
    fps: float  # of the source video
    indices: list[int]  # source frame index of each sampled frame
    images: list[np.ndarray]  # sampled frames (BGR), longest side at most FRAME_SIDE
    box: tuple[float, float, float, float] | None = None  # crop (x0, y0, x1, y1), share of the side


def check_video(info: video.VideoInfo, max_seconds: float) -> None:
    """Raise InsufficientEvidence with every reason why the video cannot be analyzed."""
    reasons = []
    seconds = info.frame_count / info.fps
    if info.fps < MIN_FPS:
        reasons.append(f"El video tiene {info.fps:.0f} cuadros por segundo; graba a 60 o 120 fps.")
    if seconds < MIN_SECONDS:
        reasons.append("El video dura menos de 1 segundo; graba el saque completo.")
    if seconds > max_seconds:
        reasons.append(f"El video dura más de {max_seconds:.0f} s; graba un solo saque por video.")
    if reasons:
        raise InsufficientEvidence(" ".join(reasons))


def sample(path, frames: int = 32, max_seconds: float = 20) -> Frames:
    """Check the video and keep `frames` evenly spaced frames. Raises InsufficientEvidence or
    ValueError."""
    info = video.probe(path)
    check_video(info, max_seconds)
    pairs = video.sample(path, frames)
    if not pairs:
        raise ValueError(f"No frames could be decoded from {path}")
    return Frames(
        video=str(path),
        fps=info.fps,
        indices=[index for index, _ in pairs],
        images=[resize(frame, FRAME_SIDE) for _, frame in pairs],
    )


def resample(frames: Frames, first: int, last: int) -> Frames:
    """As many frames as `frames` has, evenly spaced between source frames `first` and `last`:
    a closer look at one part of the video, cropped to `frames.box`."""
    pairs = video.sample(frames.video, len(frames.images), first, last)
    return replace(
        frames,
        indices=[index for index, _ in pairs],
        images=[resize(crop(frame, frames.box), FRAME_SIDE) for _, frame in pairs],
    )


def zoom(frames: Frames, box: tuple[float, float, float, float]) -> Frames:
    """The same frames, decoded again from the video and cropped to `box`, so the cropped part
    keeps the source resolution."""
    return resample(replace(frames, box=box), frames.indices[0], frames.indices[-1])


def crop(image: np.ndarray, box: tuple[float, float, float, float] | None) -> np.ndarray:
    """The part of `image` inside `box` (x0, y0, x1, y1, as shares of the width and height);
    the whole image without a box."""
    if box is None:
        return image
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    return image[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)]


def resize(image: np.ndarray, longest: int) -> np.ndarray:
    scale = longest / max(image.shape[:2])
    return (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image
    )


def label(image: np.ndarray, text: str, height: float = 0.05) -> np.ndarray:
    """Copy of the image with white, black-outlined `text` in the top-left corner.
    `height` is the text height as a share of the image height."""
    out = image.copy()
    scale = height * image.shape[0] / 22  # FONT_HERSHEY_SIMPLEX is about 22 px high at scale 1
    origin = (round(20 * scale), round(36 * scale))
    for color, width in (((0, 0, 0), 6), ((255, 255, 255), 2)):
        cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    max(1, round(width * scale)), cv2.LINE_AA)  # fmt: skip
    return out


def write_slow_motion(frames: Frames, key_frames: dict[str, int], labels: dict[str, str],
                      out: Path, slowdown: float = 4) -> Path:  # fmt: skip
    """Write the video in slow motion. Every frame shows the latest phase whose key frame
    (an index into frames.images) has already passed."""
    starts = sorted((frames.indices[i], phase) for phase, i in key_frames.items())

    def labeled():
        for index, frame in video.iter_frames(frames.video):
            current = [phase for start, phase in starts if start <= index]
            frame = crop(frame, frames.box)
            yield label(frame, labels.get(current[-1], current[-1])) if current else frame

    video.write_video(out, labeled(), frames.fps / slowdown)
    return out
