"""Pose estimation with MediaPipe Pose Landmarker (Apache-2.0): 33 landmarks per frame.

The detector sees only a fixed square around the region that moves (core.video.motion): a player
far from the camera then fills the image, and a bystander is mostly left out. Landmarks are
mapped back to pixels of the full frame.
"""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from core import ROOT, video

LANDMARKS = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear",
    "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)  # fmt: skip

SKELETON = tuple((c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS)

URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)
MODELS_DIR = ROOT / "models"  # downloaded weights; ignored by Git


def load(model: str) -> vision.PoseLandmarker:
    """model: lite | full | heavy. Downloads the weights the first time."""
    path = MODELS_DIR / f"pose_landmarker_{model}.task"
    if not path.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(v=model), partial)
        partial.rename(path)
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.VIDEO,  # tracks the person from frame to frame
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def to_frame(landmarks, box: tuple[int, int, int, int]) -> np.ndarray:
    """(33, 4) array of x, y in pixels of the full frame, z on the same scale as x, visibility.
    `landmarks` are normalized to the box."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0  # MediaPipe scales z like x
    return np.array(
        [(p.x * width + x0, p.y * height + y0, p.z * width, p.visibility) for p in landmarks],
        dtype=np.float32,
    )


def extract(path: str | Path, model: str, before: float = 1.4, after: float = 0.7) -> dict:
    """Landmarks for every frame of the video, NaN where no person was found.

    keypoints: (frames, 33, 4) from to_frame. world: (frames, 33, 3) in meters, centered on the
    hips. contact: the frame with the most motion. box: the square the detector saw.
    """
    info = video.probe(path)
    m = video.motion(path, before, after)
    x0, y0, x1, y1 = m.box
    keypoints, world = [], []
    with load(model) as landmarker:
        for index, frame in video.iter_frames(path):
            part = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(part))
            result = landmarker.detect_for_video(image, round(index * 1000 / info.fps))
            if result.pose_landmarks:
                keypoints.append(to_frame(result.pose_landmarks[0], m.box))
                world.append([(p.x, p.y, p.z) for p in result.pose_world_landmarks[0]])
            else:
                keypoints.append(np.full((len(LANDMARKS), 4), np.nan, np.float32))
                world.append(np.full((len(LANDMARKS), 3), np.nan))
    return {
        "keypoints": np.stack(keypoints),
        "world": np.array(world, dtype=np.float32),
        "fps": info.fps,
        "contact": m.peak,
        "first": m.first,
        "last": m.last,
        "box": np.array(m.box),
        "model": model,
    }


def draw(frame: np.ndarray, keypoints: np.ndarray, min_visibility: float = 0.5) -> np.ndarray:
    """A copy of the frame with the skeleton of one frame's (33, 4) keypoints drawn on it."""
    out = frame.copy()
    thickness = max(2, round(frame.shape[0] / 360))
    seen = keypoints[:, 3] >= min_visibility  # NaN compares False
    points = [tuple(int(v) for v in p) if ok else None for p, ok in zip(keypoints[:, :2], seen)]
    for a, b in SKELETON:
        if points[a] and points[b]:
            cv2.line(out, points[a], points[b], (0, 255, 0), thickness, cv2.LINE_AA)
    for point in filter(None, points):
        cv2.circle(out, point, thickness * 2, (0, 0, 255), -1, cv2.LINE_AA)
    return out
