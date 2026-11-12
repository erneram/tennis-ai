import numpy as np

from strokes.serve import body, dataset, train

J = dataset.JOINTS.index
T = dataset.BEFORE + 1 + dataset.AFTER


def serve() -> np.ndarray:
    """One window: the left arm tosses (the right wrist is highest at contact) and the player
    moves toward +x."""
    x = np.zeros((1, T, len(dataset.JOINTS), 2))
    x[:, :, [J("left_shoulder"), J("right_shoulder")], 1] = -1
    x[:, dataset.BEFORE, J("right_wrist"), 1] = -2
    x[:, :, dataset.HIPS, 0] = np.linspace(0, 1, T)[None, :, None]
    return x


def test_measures_read_the_same_for_either_tossing_arm_and_camera_side():
    x, mask = serve(), np.ones((1, T, len(dataset.JOINTS)), bool)
    mirrored = x.copy()
    train.mirror(mirrored, mask.copy(), np.array([True]))  # the right arm tosses
    flipped = x * [-1, 1]  # same arm, the court on the other side of the image
    for other in (mirrored, flipped):
        for aspect, value in body.measures(other, mask).items():
            assert np.allclose(value, body.measures(x, mask)[aspect]), aspect


def test_deeper_knee_bend_and_more_forward_landing_mean_less_to_improve():
    x, mask = serve(), np.ones((1, T, len(dataset.JOINTS)), bool)
    bent = x.copy()
    bent[:, dataset.BEFORE - 20 : dataset.BEFORE - 10, dataset.HIPS, 1] += 0.3  # hips go down
    assert body.measures(bent, mask)["legs"] < body.measures(x, mask)["legs"]
    further = x.copy()
    step = np.linspace(0, 1, T - dataset.BEFORE)[None, :, None]
    further[:, dataset.BEFORE :, dataset.HIPS, 0] += step
    assert body.measures(further, mask)["landing"] < body.measures(x, mask)["landing"]


def test_targets_are_one_for_needs_work_and_leave_out_unseen_answers():
    answered = {"v1": {"legs": "little", "shoulders": "unseen", "racket_at_peak": "trophy",
                       "racket_drop": "good", "arm_at_contact": "forward", "landing": "too_much"}}  # fmt: skip
    y = body.targets(answered, ["v1", "v2"])
    assert y.shape == (2, len(body.ANSWERS))
    assert np.array_equal(y[0], [1, np.nan, 0, 0, 1, np.nan], equal_nan=True)
    assert np.isnan(y[1]).all()
