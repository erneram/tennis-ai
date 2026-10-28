import math

import pytest

from eval.run_eval import compute, kappa, macro_prf


def test_macro_prf():
    precision, recall, f1 = macro_prf(["a", "a", "b"], ["a", "b", "b"])
    assert precision == pytest.approx(0.75)  # a: 1/1, b: 1/2
    assert recall == pytest.approx(0.75)  # a: 1/2, b: 1/1
    assert f1 == pytest.approx(2 / 3)
    assert all(math.isnan(v) for v in macro_prf([], []))


def label(valid, phase="", category="", drill_id=""):
    return {"valid": valid, "phase": phase, "category": category, "drill_id": drill_id}


def test_compute_counts_wrong_rejections_as_errors():
    answer = {
        "phase": "contact",
        "category": "low_contact_point",
        "drill_id": "higher_toss",
        "consistent": True,
    }
    records = [
        # correct answer
        {"label": label(True, "contact", "low_contact_point", "higher_toss"), "valid": True,
         "prediction": answer, "seconds": 2.0, "tokens": 6000, "mem_gb": 5.0},
        # valid video wrongly rejected
        {"label": label(True, "trophy", "toss_arm_drops_early"), "valid": False, "seconds": 1.0},
        # invalid video correctly rejected
        {"label": label(False), "valid": False, "seconds": 1.0},
    ]  # fmt: skip
    metrics = compute(records)
    assert metrics["retake_acc"] == pytest.approx(2 / 3, abs=1e-3)
    assert metrics["phase_acc"] == 0.5
    assert metrics["drill_acc"] == 1.0
    assert metrics["hallucination_rate"] == 0.0
    assert metrics["latency_s"] == 2.0
    assert metrics["tokens"] == 6000 and metrics["mem_gb"] == 5.0


def test_kappa():
    assert kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0  # chance level
    assert kappa(["a", "a", "a", "b"], ["a", "a", "b", "b"]) == pytest.approx(0.5)
    assert kappa(["a"], ["a"]) == 1.0 and math.isnan(kappa([], []))
