import numpy as np
import pytest

from eval import improvement
from strokes.serve import dataset

J = dataset.JOINTS.index


def test_toss_height_is_read_when_the_racket_wrist_is_lowest():
    x = np.zeros((1, dataset.BEFORE + 1 + dataset.AFTER, len(dataset.JOINTS), 2))
    x[0, :, J("right_wrist"), 1] = -1  # tossing wrist one torso length above its shoulder
    x[0, dataset.BEFORE - 5 :, J("right_wrist"), 1] = 0.5  # until it drops
    x[0, dataset.BEFORE, J("left_wrist"), 1] = -3  # racket wrist higher: the right arm tosses
    x[0, dataset.BEFORE - 10, J("left_wrist"), 1] = 2  # image y grows downward
    mask = np.ones(x.shape[:3], bool)
    assert improvement.toss_height(x, mask) == pytest.approx([1])
    x[0, dataset.BEFORE - 3, J("left_wrist"), 1] = 3  # racket lowest after the hand dropped
    assert improvement.toss_height(x, mask) == pytest.approx([-0.5])


def test_gap_closed_is_the_share_of_the_way_to_the_professionals():
    before, after, pros = np.array([0.2, 0.4]), np.array([0.6, 0.8]), np.array([1.0, 1.2])
    assert improvement.gap_closed(before, after, pros, np.median) == pytest.approx(0.5)
