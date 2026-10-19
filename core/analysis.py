"""End-to-end analysis of one video, up to the language model:
video -> pose -> quality checks -> phases -> measurements -> key frames.

The stroke module is passed in as an argument, so the core never imports strokes.
A stroke module provides PHASES, LABELS, segment(), key_frames() and measure()
(see strokes/base.py).
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core import ROOT, metrics, pose, video

CACHE_DIR = ROOT / "data" / "processed" / "pose_cache"
CORE_JOINTS = list(range(5, 17))  # shoulders to ankles

# Quality thresholds. ponytail: initial guesses, tune them on the first recorded videos.
MIN_FPS = 24
MIN_SECONDS = 1.0
MIN_DETECTED = 0.7  # share of frames where a person is found
MIN_CONFIDENCE = 0.5  # mean confidence of the core joints
MAX_OUTSIDE = 0.1  # share of core-joint points outside the image


class InsufficientEvidence(Exception):
    """The video does not show enough to give feedback; the player must record again."""


@dataclass
class Analysis:
    video: str
    fps: float  # of the processed sequence: source fps / step
    step: int
    frame_indices: np.ndarray  # source frame index of each processed frame
    keypoints: np.ndarray  # (frames, 17, 3) raw pose output
    xy: np.ndarray  # (frames, 17, 2) cleaned positions
    phases: dict[str, int]  # phase -> first processed frame of that phase
    key_frames: dict[str, int]  # phase -> processed frame that best shows it
    measurements: dict[str, float]
    images: dict[str, np.ndarray]  # phase -> key frame cropped around the player (BGR)
    overlays: dict[str, np.ndarray]  # phase -> same crop with the skeleton drawn
    seconds: float  # pose estimation + analysis time


def estimate_pose(path, backend: str, model: str, step: int = 1, cache: bool = True):
    """Return (info, keypoints, frame_indices, frame_shape). Cached in data/processed/pose_cache."""
    info = video.probe(path)
    path = Path(path)
    cache_file = CACHE_DIR / f"{path.stem}_{path.stat().st_size}_{backend}_{model}_{step}.npz"
    if cache and cache_file.exists():
        data = np.load(cache_file)
        return info, data["keypoints"], data["frame_indices"], tuple(data["shape"])

    detect = pose.load(backend, model)
    indices, keypoints, shape = [], [], None
    for index, frame in video.iter_frames(path, step):
        keypoints.append(detect(frame, int(index * 1000 / info.fps)))
        indices.append(index)
        shape = frame.shape[:2]
    if not keypoints:
        raise ValueError(f"No frames could be decoded from {path}")

    keypoints, indices = np.stack(keypoints), np.array(indices)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, keypoints=keypoints, frame_indices=indices, shape=shape)
    return info, keypoints, indices, shape


def check_quality(info: video.VideoInfo, keypoints: np.ndarray, shape, max_seconds: float):
    """Raise InsufficientEvidence with every reason why the video cannot be analyzed."""
    reasons = []
    seconds = info.frame_count / info.fps
    if info.fps < MIN_FPS:
        reasons.append(f"El video tiene {info.fps:.0f} cuadros por segundo; graba a 60 o 120 fps.")
    if seconds < MIN_SECONDS:
        reasons.append("El video dura menos de 1 segundo; graba el saque completo.")
    if seconds > max_seconds:
        reasons.append(f"El video dura más de {max_seconds:.0f} s; graba un solo saque por video.")

    found = keypoints[:, :, 2].max(axis=1) > 0
    core = keypoints[found][:, CORE_JOINTS]
    if found.mean() < MIN_DETECTED:
        reasons.append("No se detecta al jugador en gran parte del video.")
    elif core[..., 2].mean() < MIN_CONFIDENCE:
        reasons.append(
            "El cuerpo se ve poco: revisa la luz, la distancia y el ángulo de la cámara."
        )
    else:
        height, width = shape
        x, y = core[..., 0], core[..., 1]
        outside = (x < 0) | (x > width) | (y < 0) | (y > height)
        if outside.mean() > MAX_OUTSIDE:
            reasons.append("Parte del cuerpo queda fuera del encuadre; aleja la cámara.")

    if reasons:
        raise InsufficientEvidence(" ".join(reasons))


def analyze(path, stroke, backend: str, model: str, step: int = 1, max_seconds: float = 20,
            cache: bool = True) -> Analysis:  # fmt: skip
    """Run everything before the language model. Raises InsufficientEvidence or ValueError."""
    start = time.perf_counter()
    info, keypoints, indices, shape = estimate_pose(path, backend, model, step, cache)
    check_quality(info, keypoints, shape, max_seconds)

    fps = info.fps / step
    xy = metrics.clean(keypoints, fps)
    phases = stroke.segment(xy, fps)
    key_frames = stroke.key_frames(phases, len(xy), fps)
    measurements = stroke.measure(xy, fps, phases)
    images, overlays = _crop_key_frames(path, step, keypoints, key_frames)

    return Analysis(
        video=str(path),
        fps=fps,
        step=step,
        frame_indices=indices,
        keypoints=keypoints,
        xy=xy,
        phases=phases,
        key_frames=key_frames,
        measurements=measurements,
        images=images,
        overlays=overlays,
        seconds=time.perf_counter() - start,
    )


def _crop_box(keypoints: np.ndarray, shape) -> tuple[int, int, int, int]:
    """Box around the player with extra room above the head for the toss and the racket."""
    height, width = shape
    seen = keypoints[keypoints[:, 2] >= 0.3, :2]
    if len(seen) < 4:
        return 0, 0, width, height
    (x0, y0), (x1, y1) = seen.min(axis=0), seen.max(axis=0)
    body = max(y1 - y0, 1)
    top, bottom = y0 - 0.7 * body, y1 + 0.1 * body
    half = max((x1 - x0) * 0.8, (bottom - top) * 0.4)
    center = (x0 + x1) / 2
    box = (center - half, top, center + half, bottom)
    left, top, right, bottom = (round(v) for v in box)
    return max(left, 0), max(top, 0), min(right, width), min(bottom, height)


def _resize(image: np.ndarray, longest: int = 512) -> np.ndarray:
    scale = longest / max(image.shape[:2])
    return (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else image
    )


def _crop_key_frames(path, step: int, keypoints: np.ndarray, key_frames: dict[str, int]):
    last = max(key_frames.values())
    images, overlays = {}, {}
    for processed, (_, frame) in enumerate(video.iter_frames(path, step)):
        for phase in [p for p, f in key_frames.items() if f == processed]:
            left, top, right, bottom = _crop_box(keypoints[processed], frame.shape[:2])
            drawn = pose.draw(frame, keypoints[processed])
            images[phase] = _resize(frame[top:bottom, left:right])
            overlays[phase] = _resize(drawn[top:bottom, left:right])
        if processed >= last:
            break
    return {p: images[p] for p in key_frames}, {p: overlays[p] for p in key_frames}


def write_overlay(
    analysis: Analysis, out: Path, labels: dict[str, str], slowdown: float = 4
) -> Path:
    """Write the skeleton overlay in slow motion, labeling the current phase on every frame."""
    starts = sorted((frame, phase) for phase, frame in analysis.phases.items())

    def frames():
        for processed, (_, frame) in enumerate(video.iter_frames(analysis.video, analysis.step)):
            drawn = pose.draw(frame, analysis.keypoints[processed])
            current = [phase for start, phase in starts if start <= processed]
            if current:
                scale = frame.shape[0] / 720
                origin = (int(20 * scale), int(50 * scale))
                text = labels.get(current[-1], current[-1])
                for color, width in (((0, 0, 0), 6), ((255, 255, 255), 2)):
                    cv2.putText(drawn, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, color,
                                max(1, int(width * scale)), cv2.LINE_AA)  # fmt: skip
            yield drawn

    video.write_video(out, frames(), analysis.fps / slowdown)
    return out
