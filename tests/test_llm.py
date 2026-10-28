import numpy as np
import pytest

from core.analysis import InsufficientEvidence
from core.llm import Generation, validate
from strokes import base, serve

FEEDBACK = {"phase": "toss", "category": "toss_arm_bent", "observation": "Estira el brazo.",
            "evidence_phases": ["toss"], "drill_id": "toss_target", "confidence": "high"}  # fmt: skip


class FakeVLM:
    """Records the prompt and answers like the real model would."""

    def __init__(self, answer: dict):
        self.answer = answer

    def generate_json(self, system, user, images, schema, **generation):
        self.system, self.user, self.images = system, user, images
        validate(self.answer, schema)
        return Generation(
            self.answer, prompt_tokens=900, generation_tokens=60, seconds=2.0, peak_memory_gb=4
        )


def frames(n: int) -> list[np.ndarray]:
    return [np.zeros((720, 1280, 3), np.uint8) for _ in range(n)]


def test_locate_phases_numbers_the_frames_and_keeps_phases_in_order():
    # contact (frame 15) comes before leg_drive (frame 20): it is moved up to frame 20
    answer = {"retake": "none", "stance": 2, "toss": 6, "trophy": 12, "leg_drive": 20,
              "contact": 15, "follow_through": 25}  # fmt: skip
    vlm = FakeVLM(answer)
    key_frames, result = base.locate_phases(serve, frames(32), vlm, "v4")
    assert key_frames == {"stance": 1, "toss": 5, "trophy": 11, "leg_drive": 19,
                          "contact": 19, "follow_through": 24}  # fmt: skip
    assert result.prompt_tokens == 900
    assert len(vlm.images) == 32 and max(vlm.images[0].shape) == base.LOCATE_SIDE
    assert vlm.images[0].any()  # the frame number is drawn on the black frame
    assert "Frames: 1 to 32." in vlm.user and "not_from_the_side" in vlm.user
    assert "$" not in vlm.user


def test_locate_phases_asks_for_a_new_recording():
    answer = {"retake": "not_from_the_side", **dict.fromkeys(serve.PHASES, 1)}
    with pytest.raises(InsufficientEvidence, match="no está de lado"):
        base.locate_phases(serve, frames(8), FakeVLM(answer), "v4")


def test_phases_schema_only_allows_existing_frames():
    schema = base.phases_schema(serve, 8, ["too_dark"])
    good = {"retake": "too_dark", **dict.fromkeys(serve.PHASES, 8)}
    validate(good, schema)
    with pytest.raises(ValueError):
        validate({**good, "contact": 9}, schema)
    with pytest.raises(ValueError):
        validate({**good, "retake": "invented"}, schema)


def test_full_prompt_has_criteria_and_no_measurements():
    vlm = FakeVLM(FEEDBACK)
    images = dict(zip(serve.PHASES, frames(6)))
    result = base.feedback(serve, images, vlm, "full", "v4")
    assert "toss: toss_arm_bent:" in vlm.user and "Tossing arm straight" in vlm.user
    assert "$" not in vlm.user and len(vlm.images) == len(serve.PHASES)
    assert "Never state numbers" in vlm.system
    assert result["consistent"] and result["source"] == "vlm:full:v4"


def test_simple_baseline_prompt_has_no_criteria():
    vlm = FakeVLM(FEEDBACK)
    base.feedback(serve, dict(zip(serve.PHASES, frames(6))), vlm, "simple", "v4")
    assert "Tossing arm straight" not in vlm.user and "toss_target" in vlm.user


def test_validate_rejects_unknown_values():
    schema = base.feedback_schema(serve)
    good = {"phase": "contact", "category": "low_contact_point", "observation": "x",
            "evidence_phases": ["contact"], "drill_id": "higher_toss", "confidence": "low"}  # fmt: skip
    validate(good, schema)
    with pytest.raises(ValueError):
        validate({**good, "drill_id": "invented"}, schema)
    with pytest.raises(ValueError):
        validate({**good, "evidence_phases": ["warmup"]}, schema)
