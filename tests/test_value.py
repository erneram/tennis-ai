import numpy as np

from eval import value

CONFIG = {"prevalence": 0.5, "coach_per_hour": 120, "drill_minutes": 30, "hit": 1.0, "miss": 0.5,
          "false_alarm": 1.0, "analyses_per_month": 10, "watts": 0, "kwh_price": 0}  # fmt: skip


def test_value_of_a_perfect_model_and_of_one_that_always_says_drops_early():
    assert value.value(1, 0, CONFIG, 0) == 60 * 0.5  # a block is Q60; half the serves have it
    assert value.value(1, 1, CONFIG, 0) == 0  # every false alarm cancels a hit


def test_threshold_is_chosen_by_value():
    truth = np.array([True, True, False, False])
    scores = np.array([[0.9, 0.6, 0.4, 0.1]])
    threshold = value.best_threshold(truth, scores, CONFIG, 0)
    assert 0.4 < threshold <= 0.6
    assert value.rates(truth, scores, threshold) == ([1.0], [0.0])
