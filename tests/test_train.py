import numpy as np

from strokes.serve import dataset, pretrain, train

J = dataset.JOINTS.index


def test_mirror_swaps_left_and_right_and_undoes_itself():
    assert train.MIRROR[J("left_wrist")] == J("right_wrist")
    assert train.MIRROR[J("nose")] == J("nose")
    assert [train.MIRROR[i] for i in train.MIRROR] == list(range(len(dataset.JOINTS)))


def test_network_gives_one_logit_per_window():
    x = np.zeros((3, dataset.BEFORE + 1 + dataset.AFTER, len(dataset.JOINTS), 2))
    mask = np.ones(x.shape[:3], bool)
    assert train.network()(train.features(x, mask)).shape == (3, 1)


def test_arms_puts_the_tossing_arm_on_the_left():
    x = np.zeros((2, dataset.BEFORE + 1 + dataset.AFTER, len(dataset.JOINTS), 2))
    x[0, dataset.BEFORE, J("right_wrist"), 1] = -3  # left arm tosses: keep as is
    x[1, dataset.BEFORE, J("left_wrist"), 1] = -3  # right arm tosses: mirror
    x[:, :, J("right_elbow")] = [0.5, 0.2]
    arms, mask = train.arms(x, np.ones(x.shape[:3], bool))
    elbow = train.ARMS.index(J("left_elbow"))
    assert arms.shape[2] == mask.shape[2] == 6
    assert np.allclose(arms[0, :, elbow], 0) and np.allclose(arms[1, :, elbow], [-0.5, 0.2])


def test_pretraining_hides_stretches_and_a_straight_motion_is_filled_exactly():
    x = np.zeros((3, 64, 6, 2))
    x[..., 1] = np.linspace(-2, 1, 64)[None, :, None]  # every joint moves at constant speed
    mask = np.ones(x.shape[:3], bool)
    shown, shown_mask, hidden = pretrain.hide(x, mask, np.random.default_rng(0))
    assert hidden.any() and not shown[hidden].any() and not shown_mask[hidden].any()
    assert pretrain.interpolation(x, mask, np.random.default_rng(0)) < 1e-12
