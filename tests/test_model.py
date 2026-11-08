import numpy as np

from strokes.serve import dataset, model


def test_occlusion_replaces_one_joint_in_one_stretch():
    x = np.zeros((2, dataset.BEFORE + 1 + dataset.AFTER, 6, 2))
    reference = np.ones(x.shape[1:])
    out = model.replace(x, reference, 2, model.STARTS[4])
    changed = np.argwhere(out != 0)
    assert set(changed[:, 1]) == set(range(32, 40)) and set(changed[:, 2]) == {2}
    assert model.seconds(model.STARTS[4]) == -16 / 30 and not x.any()


def test_shoulder_width_reads_only_the_stance():
    x = np.zeros((2, dataset.BEFORE + 1 + dataset.AFTER, len(dataset.JOINTS), 2))
    x[:, :, 1, 0], x[:, :, 2, 0] = 0.3, -0.3  # from the side: shoulders apart across the image
    x[1, : model.STANCE, 2, 0] = 0.2  # from the front: one shoulder behind the other
    x[:, model.STANCE :, 2, 0] = 0.3  # after the stance the player turns: not read
    assert np.allclose(model.shoulder_width(x), [0.6, 0.1])


def test_the_app_model_has_every_seed_and_the_threshold_chosen_by_value():
    m1e = model.load("m1e")
    assert len(m1e["models"]) == 5 and m1e["heights"] and len(m1e["parts"]) == 4
    assert 0 < m1e["threshold"] < 1 and m1e["reference"].shape == (64, 4, 1)
