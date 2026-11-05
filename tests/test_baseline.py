import numpy as np

from strokes.serve import baseline, dataset

J = dataset.JOINTS.index


def test_drop_time_is_the_last_frame_the_tossing_wrist_was_above_its_shoulder():
    x = np.zeros((dataset.BEFORE + 1 + dataset.AFTER, len(dataset.JOINTS), 2), np.float32)
    x[:, [J("left_shoulder"), J("right_shoulder")], 1] = -1  # one torso above the hips
    x[dataset.BEFORE, J("right_wrist"), 1] = -3  # racket hand at contact
    x[: dataset.BEFORE - 5, J("left_wrist"), 1] = -2  # tossing hand last up 6 frames before
    assert np.isclose(baseline.drop_time(x), 6 / dataset.FPS)
    x[:, J("left_wrist"), 1] = 0  # never above the shoulder
    assert np.isclose(baseline.drop_time(x), (dataset.BEFORE + 1) / dataset.FPS)


def test_auroc_counts_ties_half():
    positive = np.array([True, True, False, False])
    assert baseline.auroc(positive, np.array([3.0, 2, 1, 2])) == 3.5 / 4
