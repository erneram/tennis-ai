"""Ultralytics YOLO Pose (AGPL-3.0, optional group: uv sync --group yolo). Outputs COCO-17 natively."""

import numpy as np
import torch
from ultralytics import YOLO

from core.pose import MODELS_DIR, Detector, missing


def load(model: str) -> Detector:
    """model: weights file name, for example yolo26n-pose.pt (downloaded on first use)."""
    MODELS_DIR.mkdir(exist_ok=True)
    net = YOLO(str(MODELS_DIR / model))
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # ~3x faster than CPU on M5
    net(np.zeros((640, 640, 3), np.uint8), device=device, verbose=False)  # warm up, keeps timings fair

    def detect(frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        result = net(frame, device=device, verbose=False)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return missing()
        # ponytail: largest box = the player; use model.track() IDs if other people appear in frame.
        boxes = result.boxes.xywh.cpu().numpy()
        player = int(np.argmax(boxes[:, 2] * boxes[:, 3]))
        return result.keypoints.data[player].cpu().numpy().astype(float)

    return detect
