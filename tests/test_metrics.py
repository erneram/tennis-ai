import numpy as np
import pytest

from core import metrics
from core.analysis import InsufficientEvidence, check_quality
from core.video import VideoInfo


def test_angle_and_missing_points():
    a, b, c = np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0])
    assert metrics.angle(a, b, c) == pytest.approx(90)
    assert np.isnan(metrics.angle(a, b, np.array([np.nan, 1.0])))


def test_clean_fills_gaps_and_keeps_unseen_joints_nan():
    keypoints = np.zeros((5, 17, 3))
    keypoints[:, 0] = [
        (0, 0, 1),
        (1, 1, 1),
        (9, 9, 0.0),
        (3, 3, 1),
        (4, 4, 1),
    ]  # frame 2 unreliable
    xy = metrics.clean(keypoints, fps=10)  # 10 fps -> no smoothing
    assert xy[2, 0].tolist() == [2, 2]
    assert np.isnan(xy[:, 1]).all()


def test_quality_rejects_short_low_fps_video_without_player():
    keypoints = np.zeros((10, 17, 3))
    with pytest.raises(InsufficientEvidence) as error:
        check_quality(
            VideoInfo(fps=15, frame_count=10, width=64, height=64), keypoints, (64, 64), 20
        )
    message = str(error.value)
    assert (
        "15 cuadros" in message and "menos de 1 segundo" in message and "No se detecta" in message
    )
