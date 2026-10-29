import numpy as np
import pytest

from core import video
from core.analysis import InsufficientEvidence, sample
from core.llm import Generation, validate
from strokes import base, serve

FEEDBACK = {"seen": "Toss: arm bent.", "phase": "toss", "category": "toss_arm_bent",
            "observation": "Estira el brazo.", "evidence_phases": ["toss"],
            "drill_id": "toss_target", "confidence": "high"}  # fmt: skip
NO_PROBLEM = {**FEEDBACK, "phase": "none", "category": "no_clear_problem",
              "drill_id": "steady_serves", "confidence": "medium"}  # fmt: skip


class FakeVLM:
    """Records the last prompt and gives the answers in order, like the real model would."""

    def __init__(self, *answers: dict):
        self.answers = list(answers)

    def generate_json(self, system, user, images, schema, **generation):
        self.system, self.user, self.images = system, user, images
        answer = self.answers.pop(0)
        validate(answer, schema)
        return Generation(
            answer, prompt_tokens=900, generation_tokens=60, seconds=2.0, peak_memory_gb=4
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


def phases(*numbers: int) -> dict:
    return dict(zip(serve.PHASES, numbers, strict=True))


def test_v5_finds_the_serve_then_samples_inside_it(tmp_path):
    path = tmp_path / "serve.mp4"
    video.write_video(path, [np.full((64, 96, 3), i, np.uint8) for i in range(40)], fps=30)
    frames = sample(path, frames=8)  # source frames 0, 6, 11, 17, 22, 28, 33, 39
    vlm = FakeVLM({"retake": "none", "start": 3, "end": 6}, phases(1, 2, 4, 5, 7, 8))
    frames, key_frames, calls = base.phase_frames(serve, frames, vlm, "v5")
    # one frame of margin on each side: source frames 6 to 33
    assert frames.indices[0] == 6 and frames.indices[-1] == 33 and len(frames.images) == 8
    assert key_frames == phases(0, 1, 3, 4, 6, 7) and len(calls) == 2
    assert "Frames: 1 to 8." in vlm.user and "$" not in vlm.user


def test_v5_rejects_six_frames_in_a_row(tmp_path):
    path = tmp_path / "serve.mp4"
    video.write_video(path, [np.zeros((64, 96, 3), np.uint8)] * 40, fps=30)
    vlm = FakeVLM({"retake": "none", "start": 1, "end": 8}, phases(1, 2, 3, 4, 5, 6))
    with pytest.raises(ValueError, match="guess"):
        base.phase_frames(serve, sample(path, frames=8), vlm, "v5")


def test_v5_asks_for_a_new_recording_before_looking_for_phases():
    vlm = FakeVLM({"retake": "serve_incomplete", "start": 1, "end": 1})
    with pytest.raises(InsufficientEvidence, match="saque completo"):
        base.find_window(serve, frames(8), vlm, "v5")
    assert "bouncing is not part of the serve" in vlm.system


def test_no_clear_problem_is_a_consistent_answer():
    vlm = FakeVLM(NO_PROBLEM)
    result = base.feedback(serve, dict(zip(serve.PHASES, frames(6))), vlm, "full", "v5")
    assert result["consistent"] and result["seen"] == "Toss: arm bent."
    assert "no_clear_problem" in vlm.user and "steady_serves" in vlm.user


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
    good = {"seen": "x", "phase": "contact", "category": "low_contact_point", "observation": "x",
            "evidence_phases": ["contact"], "drill_id": "higher_toss", "confidence": "low"}  # fmt: skip
    validate(good, schema)
    with pytest.raises(ValueError):
        validate({**good, "drill_id": "invented"}, schema)
    with pytest.raises(ValueError):
        validate({**good, "evidence_phases": ["warmup"]}, schema)
