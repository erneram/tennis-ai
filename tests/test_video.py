import numpy as np
import pytest

from core import analysis, video
from core.analysis import InsufficientEvidence, check_video
from core.video import VideoInfo


def clip(path, n=10):
    frames = [np.full((64, 96, 3), i % 10 * 20, dtype=np.uint8) for i in range(n)]
    assert video.write_video(path, frames, fps=30) == n
    return path


def test_video_roundtrip_and_step(tmp_path):
    path = clip(tmp_path / "clip.mp4")
    info = video.probe(path)
    assert (info.frame_count, info.width, info.height) == (10, 96, 64)
    assert round(info.fps) == 30
    assert [i for i, _ in video.iter_frames(path, step=3)] == [0, 3, 6, 9]


def test_sample_spreads_frames_over_the_whole_video(tmp_path):
    path = clip(tmp_path / "clip.mp4", n=40)
    assert [i for i, _ in video.sample(path, 4)] == [0, 13, 26, 39]
    assert len(video.sample(path, 100)) == 40


def test_check_video_rejects_short_low_fps_video():
    with pytest.raises(InsufficientEvidence) as error:
        check_video(VideoInfo(fps=15, frame_count=10, width=64, height=64), 20)
    assert "15 cuadros" in str(error.value) and "menos de 1 segundo" in str(error.value)


def test_slow_motion_labels_each_frame_after_its_phase(tmp_path):
    path = clip(tmp_path / "clip.mp4", n=40)
    frames = analysis.Frames(str(path), 30, [0, 13, 26, 39], [])
    out = analysis.write_slow_motion(frames, {"toss": 1, "contact": 2}, {}, tmp_path / "slow.mp4")
    written = [frame for _, frame in video.iter_frames(out)]
    assert len(written) == 40
    original = [frame for _, frame in video.iter_frames(path)]
    assert np.abs(written[5].astype(int) - original[5]).max() < 20  # before toss: no label
    assert np.abs(written[20].astype(int) - original[20]).max() > 100  # label drawn
