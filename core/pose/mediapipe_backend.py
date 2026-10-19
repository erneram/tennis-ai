"""MediaPipe Pose Landmarker (Apache-2.0), mapped from its 33 landmarks to COCO-17."""

import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from core.pose import MODELS_DIR, Detector, missing

URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)

# MediaPipe landmark index for each COCO-17 keypoint, in KEYPOINTS order.
COCO_FROM_MEDIAPIPE = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def load(model: str) -> Detector:
    """model: lite | full | heavy."""
    path = MODELS_DIR / f"pose_landmarker_{model}.task"
    if not path.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(v=model), partial)
        partial.rename(path)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return missing()
        height, width = frame.shape[:2]
        landmarks = result.pose_landmarks[0]
        return np.array(
            [
                (landmarks[i].x * width, landmarks[i].y * height, landmarks[i].visibility or 0.0)
                for i in COCO_FROM_MEDIAPIPE
            ]
        )

    return detect
