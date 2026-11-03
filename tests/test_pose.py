from types import SimpleNamespace

import numpy as np

from core import pose


def test_to_frame_maps_landmarks_from_the_box_to_the_frame():
    point = SimpleNamespace(x=0.5, y=0.25, z=-0.1, visibility=0.9)
    keypoints = pose.to_frame([point] * len(pose.LANDMARKS), (100, 200, 300, 600))
    assert keypoints.shape == (33, 4)
    assert np.allclose(keypoints[0], (200, 300, -20, 0.9))


def test_draw_skips_missing_landmarks():
    frame = np.zeros((64, 96, 3), np.uint8)
    keypoints = np.full((33, 4), np.nan, np.float32)
    assert np.array_equal(pose.draw(frame, keypoints), frame)
    keypoints[[11, 13]] = [(10, 10, 0, 0.9), (50, 40, 0, 0.9)]  # left shoulder, left elbow
    assert pose.draw(frame, keypoints).any()
