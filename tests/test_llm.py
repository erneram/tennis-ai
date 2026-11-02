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


def check(answer: str) -> dict:
    return {"seen": "Arm visible.", "answer": answer}


def test_v6_picks_the_earliest_phase_with_a_clear_no():
    # one check per category, in criteria order: toss, trophy, leg_drive, contact x2, follow_through
    vlm = FakeVLM(check("yes"), check("unclear"), check("no"), check("no"), check("yes"),
                  check("yes"), {"observation": "Flexiona más las rodillas."})  # fmt: skip
    result = base.feedback(serve, dict(zip(serve.PHASES, frames(6))), vlm, "full", "v6")
    assert (result["phase"], result["category"], result["drill_id"]) == (
        "leg_drive", "insufficient_knee_bend", "knee_load")  # fmt: skip
    assert result["confidence"] == "medium" and result["evidence_phases"] == ["leg_drive"]
    assert len(result["checks"]) == 6 and result["prompt_tokens"] == 7 * 900
    assert len(vlm.images) == 1 and "Arm visible." in vlm.user  # the observation call


def test_v6_without_a_clear_no_has_no_clear_problem():
    vlm = FakeVLM(*[check("yes")] * 5, check("unclear"))
    result = base.feedback(serve, dict(zip(serve.PHASES, frames(6))), vlm, "full", "v6")
    assert (result["phase"], result["category"], result["drill_id"]) == (
        "none", "no_clear_problem", "steady_serves")  # fmt: skip
    assert result["confidence"] == "low" and result["consistent"] and not vlm.answers
    assert "Question: Is the player balanced" in vlm.user and "$" not in vlm.user


def test_v7_maps_the_bad_state_to_a_problem():
    # toss good, trophy unclear, knees bad (problem), contact x2 good, follow_through good
    states = [c["choose"] for c in base.load(serve, "criteria.yaml")["categories"].values()
              if "choose" in c]  # fmt: skip
    picks = [states[0]["good"], "unclear", states[2]["bad"], *(s["good"] for s in states[3:])]
    vlm = FakeVLM(*map(check, picks), {"observation": "Flexiona más las rodillas."})
    result = base.feedback(serve, dict(zip(serve.PHASES, frames(6))), vlm, "full", "v7")
    assert result["category"] == "insufficient_knee_bend"
    answers = [c["answer"] for c in result["checks"].values()]
    assert answers == ["yes", "unclear", "no", "yes", "yes", "yes"]
    assert result["evidence_phases"] == ["leg_drive"] and not vlm.answers


def test_reach_grows_the_player_box_to_the_stroke():
    # player 30 % of the frame height (324 px): grown 324 px up and to each side, 15 % down
    box = base.reach([600, 500, 650, 800], {"up": 1.0, "down": 0.15, "side": 1.0}, 1920, 1080)
    assert box == pytest.approx((0.43125, 0.2, 0.81875, 0.845))
    assert base.reach([500, 500, 400, 800], {"up": 1, "down": 0, "side": 1}, 1920, 1080) is None
    assert base.reach([0, 0, 10, 10], {"up": 1, "down": 0, "side": 1}, 1920, 1080) is None


def test_v8_crops_around_the_player_before_looking_for_the_serve(tmp_path):
    path = tmp_path / "serve.mp4"
    video.write_video(path, [np.full((64, 96, 3), i, np.uint8) for i in range(40)], fps=30)
    player = {"bbox_2d": [450, 750, 550, 1000]}  # 16 px tall, on the first, middle, last frames
    vlm = FakeVLM(player, player, player, {"retake": "none", "start": 3, "end": 6},
                  phases(1, 2, 4, 5, 7, 8))  # fmt: skip
    frames, _, calls = base.phase_frames(serve, sample(path, frames=8), vlm, "v8")
    # grown 16 px up and to each side, 2.4 px down (cut at the bottom): x 27 to 69, y 32 to 64
    assert frames.box == pytest.approx((0.45 - 1 / 6, 0.5, 0.55 + 1 / 6, 1.0))
    assert frames.images[0].shape[:2] == (32, 42) and len(calls) == 5
    assert frames.indices[0] == 6 and frames.indices[-1] == 33 and len(frames.images) == 8


def test_v8_does_not_crop_when_the_box_keeps_most_of_the_frame():
    # one wrong box (the whole frame) makes the joined box too large: no crop
    vlm = FakeVLM({"bbox_2d": [450, 750, 550, 1000]}, {"bbox_2d": [0, 0, 1000, 1000]},
                  {"bbox_2d": [450, 750, 550, 1000]})  # fmt: skip
    box, results = base.find_player(serve, frames(32), vlm, "v8")
    assert box is None and len(results) == 3


def test_v8_keeps_a_check_only_when_both_questions_agree():
    # each category asks check (yes = good) then problem (yes = problem)
    answers = [("yes", "no"),  # toss: good
               ("yes", "yes"),  # trophy: the two disagree
               ("no", "yes"),  # leg_drive: problem
               ("no", "no"),  # contact, arm: disagree
               ("yes", "no"),  # contact, height: good
               ("unclear", "no")]  # follow_through: unclear  # fmt: skip
    vlm = FakeVLM(*(check(a) for pair in answers for a in pair),
                  {"observation": "Flexiona más las rodillas."})  # fmt: skip
    result = base.feedback(serve, dict(zip(serve.PHASES, frames(6))), vlm, "full", "v8")
    assert result["category"] == "insufficient_knee_bend" and not vlm.answers
    assert [c["answer"] for c in result["checks"].values()] == [
        "yes", "unclear", "no", "unclear", "yes", "unclear"]  # fmt: skip
    assert result["checks"]["insufficient_knee_bend"]["votes"] == {"check": "no", "problem": "no"}
    assert result["prompt_tokens"] == 13 * 900
