import numpy as np

from strokes.serve import dataset, train

J = dataset.JOINTS.index


def test_mirror_swaps_left_and_right_and_undoes_itself():
    assert train.MIRROR[J("left_wrist")] == J("right_wrist")
    assert train.MIRROR[J("nose")] == J("nose")
    assert [train.MIRROR[i] for i in train.MIRROR] == list(range(len(dataset.JOINTS)))


def test_network_gives_one_logit_per_window():
    x = np.zeros((3, dataset.BEFORE + 1 + dataset.AFTER, len(dataset.JOINTS), 2))
    mask = np.ones(x.shape[:3], bool)
    assert train.network()(train.features(x, mask)).shape == (3, 1)
