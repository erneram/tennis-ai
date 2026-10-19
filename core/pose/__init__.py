"""Pose estimation: MediaPipe Pose and YOLO Pose behind one common interface.

Every backend returns, per frame, an array of shape (17, 3) in COCO-17 keypoint order:
x and y in pixels, then a confidence in [0, 1]. A missing detection is NaN with confidence 0.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)  # fmt: skip

SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)  # fmt: skip

# Downloaded weights; ignored by Git.
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

Detector = Callable[[np.ndarray, int], np.ndarray]


def missing() -> np.ndarray:
    keypoints = np.full((len(KEYPOINTS), 3), np.nan)
    keypoints[:, 2] = 0.0
    return keypoints


def load(backend: str, model: str) -> Detector:
    """Return detect(frame_bgr, timestamp_ms) -> (17, 3) keypoints for the chosen backend."""
    if backend == "mediapipe":
        from core.pose import mediapipe_backend as module
    elif backend == "yolo":
        from core.pose import yolo_backend as module  # AGPL-3.0, optional: uv sync --group yolo
    else:
        raise ValueError(f"Unknown pose backend: {backend!r}")
    return module.load(model)


def draw(frame: np.ndarray, keypoints: np.ndarray, min_conf: float = 0.5) -> np.ndarray:
    """Return a copy of the frame with the skeleton drawn on it."""
    out = frame.copy()
    thickness = max(2, round(frame.shape[0] / 360))
    visible = keypoints[:, 2] >= min_conf
    points = [tuple(int(v) for v in p) if ok else None for p, ok in zip(keypoints[:, :2], visible)]
    for a, b in SKELETON:
        if points[a] and points[b]:
            cv2.line(out, points[a], points[b], (0, 255, 0), thickness, cv2.LINE_AA)
    for point in filter(None, points):
        cv2.circle(out, point, thickness * 2, (0, 0, 255), -1, cv2.LINE_AA)
    return out
