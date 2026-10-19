import numpy as np

from core import pose, video


def test_video_roundtrip_and_step(tmp_path):
    path = tmp_path / "clip.mp4"
    frames = [np.full((64, 96, 3), i * 20, dtype=np.uint8) for i in range(10)]
    assert video.write_video(path, frames, fps=30) == 10

    info = video.probe(path)
    assert (info.frame_count, info.width, info.height) == (10, 96, 64)
    assert round(info.fps) == 30
    assert [i for i, _ in video.iter_frames(path, step=3)] == [0, 3, 6, 9]


def test_draw_skips_missing_keypoints():
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    assert np.array_equal(pose.draw(frame, pose.missing()), frame)

    keypoints = pose.missing()
    keypoints[[5, 7]] = [(10, 10, 0.9), (50, 40, 0.9)]  # left shoulder -> left elbow
    assert pose.draw(frame, keypoints).any()
