import shutil

import cv2
import numpy as np
import pytest

from core import pose, video
from strokes.serve import split


def test_one_candidate_per_toss_at_the_highest_wrist():
    joint = split.JOINT
    points = np.full((30 * split.RATE, len(pose.LANDMARKS), 2), 100.0)  # 30 s, torso 100 px
    points[:, [joint["left_hip"], joint["right_hip"]], 1] = 200
    points[:, [joint["left_wrist"], joint["right_wrist"]], 1] = 180
    wrist = points[:, joint["left_wrist"], 1]
    wrist[75:90] = wrist[300:315] = 40  # two tosses of 1 s, 0.6 torso lengths up
    wrist[85] = wrist[310] = 0  # their tops
    wrist[321:330] = 40  # 0.6 s up within APART of the second toss: the same serve
    wrist[400:403] = 40  # 0.2 s: an arm raised, not a toss
    points[200:210] = np.nan  # no one found
    frames = np.arange(len(points)) * 4  # a 60 fps video sampled at 15
    found = split.candidates(60.0, frames, points)
    assert [c["frame"] for c in found] == [85 * 4, 310 * 4]
    assert found[0]["seconds"] == round(85 * 4 / 60, 3) and found[0]["height"] == 1.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_retime_plays_a_slowed_span_at_real_speed(tmp_path, monkeypatch):
    source = tmp_path / "slow.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
    for i in range(90):  # 3 s
        writer.write(np.full((64, 64, 3), i * 2, np.uint8))
    writer.release()
    monkeypatch.setattr(split, "RAW", tmp_path)
    split.retime(source, "pro09", 4.0, [(0.0, 2.0)])
    info = video.probe(tmp_path / "pro09-s01.mp4")
    assert abs(info.frame_count / info.fps - 0.5) < 0.1 and info.fps == 60  # 2 s at 4x: 0.5 s
    log = (tmp_path / "retimed.csv").read_text().splitlines()
    assert log[1] == "pro09-s01.mp4,slow.mp4,0.000,2.000,4"
