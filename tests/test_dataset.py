import numpy as np

from core import pose
from strokes.serve import dataset

SHOULDERS = [pose.LANDMARKS.index("left_shoulder"), pose.LANDMARKS.index("right_shoulder")]
HIPS = [pose.LANDMARKS.index("left_hip"), pose.LANDMARKS.index("right_hip")]
LEFT_WRIST, RIGHT_WRIST = pose.LANDMARKS.index("left_wrist"), pose.LANDMARKS.index("right_wrist")


def still_player(frames: int) -> np.ndarray:
    """Shoulders at y=100 and hips at y=200 (a torso of 100 px), wrists at the hips."""
    keypoints = np.zeros((frames, 33, 4), np.float32)
    keypoints[..., 3] = 0.9
    keypoints[:, SHOULDERS, 1] = 100
    keypoints[:, HIPS + [LEFT_WRIST, RIGHT_WRIST], 1] = 200
    return keypoints


def test_contact_is_the_highest_wrist_in_the_second_after_the_toss():
    keypoints = still_player(120)
    keypoints[20:50, LEFT_WRIST, 1] = 40  # toss: 0.6 torso lengths above the shoulders
    keypoints[60:63, RIGHT_WRIST, 1] = [20, 0, 20]  # racket arm at contact
    keypoints[100:102, RIGHT_WRIST, 1] = -50  # an arm raised more than a second later
    assert dataset.contact(keypoints, fps=30) == 61


def test_window_resamples_centers_on_the_hips_and_masks():
    keypoints = still_player(120)
    keypoints[..., 0] = np.arange(120)[:, None] * 2  # the player moves 2 px a frame at 60 fps
    keypoints[:, pose.LANDMARKS.index("nose"), 3] = 0.1  # never seen
    x, mask = dataset.window(keypoints, fps=60, contact=40)
    assert x.shape == (dataset.BEFORE + 1 + dataset.AFTER, len(dataset.JOINTS), 2)
    # contact is at 0.67 s: the first 28 frames at 30 fps (1.6 s before contact) are before the video
    assert not mask[:28].any() and mask[28:, 1:].all() and not mask[:, 0].any()
    assert np.allclose(x[dataset.BEFORE, dataset.HIPS], 0)
    assert np.allclose(x[dataset.BEFORE, dataset.SHOULDERS], (0, -1))  # one torso above the hips
    assert np.allclose(np.diff(x[28:, 1, 0]), 0.04)  # 4 px per frame at 30 fps, in torso lengths
