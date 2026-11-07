import numpy as np

from strokes.serve import dataset, model


def test_occlusion_replaces_one_joint_in_one_stretch():
    x = np.zeros((2, dataset.BEFORE + 1 + dataset.AFTER, 6, 2))
    reference = np.ones(x.shape[1:])
    out = model.replace(x, reference, 2, model.STARTS[4])
    changed = np.argwhere(out != 0)
    assert set(changed[:, 1]) == set(range(32, 40)) and set(changed[:, 2]) == {2}
    assert model.seconds(model.STARTS[4]) == -16 / 30 and not x.any()


def test_the_app_model_has_every_seed_and_the_threshold_chosen_by_value():
    m1c = model.load("m1c")
    assert len(m1c["models"]) == 5 and m1c["arms"]
    assert 0 < m1c["threshold"] < 1 and m1c["reference"].shape == (64, 6, 2)
