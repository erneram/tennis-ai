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


def test_session_folds_keep_each_session_on_one_side():
    videos = [f"ea-{s}-s{i:02d}" for s in range(1, 7) for i in range(1, 6)] + ["pro02_s01"]
    fold = body.session_folds(videos, seed=0)
    sessions = [body.SESSION.sub("", v) for v in videos]
    assert all(len({f for f, s2 in zip(fold, sessions) if s2 == s}) == 1 for s in set(sessions))
    assert len(set(fold)) == train.FOLDS


def test_m3_has_one_output_per_aspect_and_skips_unanswered_ones():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, T, len(dataset.JOINTS), 2)).astype(np.float32)
    mask = np.ones(x.shape[:3], bool)
    y = np.array([[1, np.nan], [0, np.nan], [1, 0], [0, 1], [1, 1], [0, 0]], np.float32)
    model, curve = body.fit(x, mask, y, seed=0, epochs=2)
    assert body.probability(model, x, mask).shape == (6, 2)
    assert np.isfinite(curve).all()  # the unanswered (NaN) targets never reach the loss

