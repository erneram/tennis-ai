"""Joint angles and generic, stroke-agnostic measurements on COCO-17 keypoint sequences.

Image coordinates: x grows to the right, y grows downwards (a smaller y is higher).
"""

import numpy as np

from core.pose import KEYPOINTS

INDEX = {name: i for i, name in enumerate(KEYPOINTS)}


def clean(keypoints: np.ndarray, fps: float, min_conf: float = 0.3) -> np.ndarray:
    """(frames, 17, 3) keypoints -> (frames, 17, 2) positions.

    Drops low-confidence points, fills gaps by linear interpolation over time,
    and smooths with a ~40 ms moving average. Joints never seen stay NaN.
    """
    xy = keypoints[..., :2].astype(float).copy()
    xy[keypoints[..., 2] < min_conf] = np.nan
    t = np.arange(len(xy))
    for joint in range(xy.shape[1]):
        for axis in range(2):
            series = xy[:, joint, axis]
            seen = ~np.isnan(series)
            if seen.sum() >= 2:
                xy[:, joint, axis] = np.interp(t, t[seen], series[seen])

    window = max(1, round(fps * 0.04)) | 1  # odd number of frames
    if window > 1 and len(xy) >= window:
        pad = window // 2
        padded = np.pad(xy, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        kernel = np.ones(window) / window
        xy = np.apply_along_axis(lambda s: np.convolve(s, kernel, mode="valid"), 0, padded)
    return xy


def point(xy: np.ndarray, name: str) -> np.ndarray:
    """(frames, 2) trajectory of one keypoint."""
    return xy[:, INDEX[name]]


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for points of shape (..., 2). NaN where a point is missing."""
    ba, bc = a - b, c - b
    norms = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (ba * bc).sum(axis=-1) / norms
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def torso_length(xy: np.ndarray) -> float:
    """Median distance between the shoulder and hip midpoints, used to normalize distances."""
    shoulders = (point(xy, "left_shoulder") + point(xy, "right_shoulder")) / 2
    hips = (point(xy, "left_hip") + point(xy, "right_hip")) / 2
    return float(np.nanmedian(np.linalg.norm(shoulders - hips, axis=-1)))
