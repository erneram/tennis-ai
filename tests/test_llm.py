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


WHOLE_FRAME = {"bbox_2d": [0, 0, 1000, 1000]}  # keeps the whole frame: no crop


def serve_video(tmp_path, frames=8):
    path = tmp_path / "serve.mp4"
    video.write_video(path, [np.full((64, 96, 3), i, np.uint8) for i in range(40)], fps=30)
    return sample(path, frames=frames)


def sources(frames, key_frames) -> list[int]:
    return [frames.indices[i] for i in key_frames.values()]


def test_v9_looks_closer_at_the_contact_and_the_landing(tmp_path):
    # window frames 2 to 7 with one frame of margin: source frames 6, 10, 14, 18, 21, 25, 29, 33
    vlm = FakeVLM(WHOLE_FRAME, WHOLE_FRAME, WHOLE_FRAME, {"retake": "none", "start": 3, "end": 6},
                  phases(1, 2, 4, 5, 7, 8),  # contact 29, follow_through 33
                  {"frame": 9},  # contact among 21, 22, 23, 24, 25, 26, 28, 29, 30, ...: 30
                  {"frame": 2})  # follow_through among 30, 31, 32, 33: 31  # fmt: skip
    frames, key_frames, calls = base.phase_frames(serve, serve_video(tmp_path), vlm, "v9")
    assert sources(frames, key_frames) == [6, 10, 18, 21, 30, 31] and len(calls) == 7
    assert frames.indices == sorted(frames.indices) and len(frames.images) == 10
    assert "Frames: 1 to 4." in vlm.user and "back on the ground" in vlm.user


def test_v9_tries_again_on_other_frames_after_a_guess(tmp_path):
    vlm = FakeVLM(WHOLE_FRAME, WHOLE_FRAME, WHOLE_FRAME, {"retake": "none", "start": 3, "end": 6},
                  phases(1, 2, 3, 4, 5, 6),  # a guess
                  phases(1, 1, 2, 3, 4, 4),  # on source frames 6, 14, 21, 29
                  {"frame": 5}, {"frame": 3})  # contact 25 (of 21 to 29), landing 27 (of 25 to 33)  # fmt: skip
    frames, key_frames, calls = base.phase_frames(serve, serve_video(tmp_path), vlm, "v9")
    assert sources(frames, key_frames) == [6, 6, 14, 21, 25, 27] and len(calls) == 8


def test_v9_still_fails_after_two_guesses(tmp_path):
    # 16 frames: the second try sees 8, enough for another six in a row
    vlm = FakeVLM(WHOLE_FRAME, WHOLE_FRAME, WHOLE_FRAME, {"retake": "none", "start": 3, "end": 9},
                  phases(1, 2, 3, 4, 5, 6), phases(1, 2, 3, 4, 5, 6))  # fmt: skip
    with pytest.raises(ValueError, match="guess"):
        base.phase_frames(serve, serve_video(tmp_path, frames=16), vlm, "v9")
    assert not vlm.answers


def test_v18_tries_a_third_time_on_every_third_frame(tmp_path):
    vlm = FakeVLM(WHOLE_FRAME, WHOLE_FRAME, WHOLE_FRAME, {"retake": "none", "start": 3, "end": 9},
                  phases(1, 2, 3, 4, 5, 6), phases(1, 2, 3, 4, 5, 6),  # two guesses
                  phases(1, 1, 2, 3, 5, 6),  # on every 3rd of the 16 frames
                  {"frame": 1})  # fmt: skip
    frames, key_frames, calls = base.phase_frames(serve, serve_video(tmp_path, frames=16), vlm,
                                                  "v18")  # fmt: skip
    # contact refined to source 11; leg drive 0.1 s before it, landing 0.6 s after
    assert sources(frames, key_frames) == [3, 3, 7, 8, 11, 29] and len(calls) == 8


def test_v10_looks_closer_at_the_contact_then_the_leg_drive(tmp_path):
    vlm = FakeVLM(WHOLE_FRAME, WHOLE_FRAME, WHOLE_FRAME, {"retake": "none", "start": 3, "end": 6},
                  phases(1, 2, 4, 5, 7, 8),  # trophy 18, leg_drive 21, contact 29
                  {"frame": 9},  # contact among 21 to 33: 30
                  {"frame": 6})  # leg_drive among 18, 19, 20, 21, 22, 23, ... 30: 23  # fmt: skip
    frames, key_frames, calls = base.phase_frames(serve, serve_video(tmp_path), vlm, "v10")
    assert sources(frames, key_frames) == [6, 10, 18, 23, 30, 33] and len(calls) == 7
    assert "pushes up" in vlm.user


def test_v13_places_the_leg_drive_a_tenth_of_a_second_before_contact(tmp_path):
    vlm = FakeVLM(WHOLE_FRAME, WHOLE_FRAME, WHOLE_FRAME, {"retake": "none", "start": 3, "end": 6},
                  phases(1, 2, 4, 5, 7, 8),  # trophy 18, leg_drive 21, contact 29
                  {"frame": 9})  # contact among 21 to 33: 30; leg_drive 30 - 3 frames at 30 fps  # fmt: skip
    frames, key_frames, calls = base.phase_frames(serve, serve_video(tmp_path), vlm, "v13")
    assert sources(frames, key_frames) == [6, 10, 18, 27, 30, 33] and len(calls) == 6
    assert frames.indices == sorted(frames.indices) and not vlm.answers


def test_v17_places_the_landing_after_contact_or_at_the_end_of_the_video(tmp_path):
    vlm = FakeVLM(WHOLE_FRAME, WHOLE_FRAME, WHOLE_FRAME, {"retake": "none", "start": 3, "end": 6},
                  phases(1, 2, 4, 5, 7, 8), {"frame": 9})  # contact 30  # fmt: skip
    frames, key_frames, _ = base.phase_frames(serve, serve_video(tmp_path), vlm, "v17")
    # 0.6 s after contact is frame 48: the video ends at frame 39
    assert sources(frames, key_frames) == [6, 10, 18, 27, 30, 39]


def test_v10_judges_the_tossing_arm_on_the_leg_drive_frame():
    images = {p: np.full((8, 8, 3), i, np.uint8) for i, p in enumerate(serve.PHASES)}
    answers = [("yes", "no"), ("no", "yes")] + [("yes", "no")] * 4  # the tossing arm dropped
    vlm = FakeVLM(
        *(check(a) for pair in answers for a in pair), {"observation": "Sostén el brazo."}
    )
    result = base.feedback(serve, images, vlm, "full", "v10")
    assert (result["phase"], result["category"]) == ("trophy", "toss_arm_drops_early")
    assert result["checks"]["toss_arm_drops_early"]["phase"] == "leg_drive"
    assert result["evidence_phases"] == ["leg_drive"] and vlm.images[0].max() == 3


def test_v11_asks_where_the_tossing_hand_is_on_the_upper_body():
    images = {p: np.full((80, 60, 3), i, np.uint8) for i, p in enumerate(serve.PHASES)}
    hand = {
        "seen": "The left hand is level with the shoulder.",
        "answer": "at shoulder height or below",
    }
    vlm = FakeVLM(
        check("yes"), check("no"),  # toss_arm_bent on the trophy frame: looks right
        {"bbox_2d": [250, 100, 750, 900]}, hand,  # toss_arm_drops_early: box, then the question
        *(check(a) for _ in range(4) for a in ("yes", "no")),
        {"observation": "Sostén el brazo arriba hasta que baje la raqueta."},
    )  # fmt: skip
    result = base.feedback(serve, images, vlm, "full", "v11")
    assert (result["phase"], result["category"]) == ("trophy", "toss_arm_drops_early")
    assert result["checks"]["toss_arm_drops_early"]["votes"] == {"question": "no"}
    assert result["evidence_phases"] == ["leg_drive"]
    # the observation sees what the question saw: the upper body of the leg-drive frame, enlarged
    assert vlm.images[0].shape[0] == base.FRAME_SIDE and vlm.images[0].max() == 3


def test_v11_counts_a_hand_between_shoulder_and_head_as_unclear():
    images = {p: np.zeros((80, 60, 3), np.uint8) for p in serve.PHASES}
    hand = {"seen": "The hand is at the height of the ear.",
            "answer": "between the shoulder and the top of the head"}  # fmt: skip
    vlm = FakeVLM(check("yes"), check("no"), {"bbox_2d": [0, 0, 1000, 1000]}, hand,
                  *(check(a) for _ in range(4) for a in ("yes", "no")))  # fmt: skip
    result = base.feedback(serve, images, vlm, "full", "v11")
    assert result["category"] == "no_clear_problem" and result["confidence"] == "low"
    assert result["checks"]["toss_arm_drops_early"]["answer"] == "unclear"


class SeedVLM(FakeVLM):
    """FakeVLM that also records the seed of each call."""

    def generate_json(self, system, user, images, schema, **generation):
        self.seeds = [*getattr(self, "seeds", []), generation.get("seed")]
        return super().generate_json(system, user, images, schema, **generation)


def hand_tries(*answers):
    """Box and answer of each try of the tossing-hand question."""
    box = {"bbox_2d": [0, 0, 1000, 1000]}
    return [x for a in answers for x in (box, {"seen": "The hand.", "answer": a})]


def test_v19_asks_the_tossing_hand_three_times_and_keeps_the_majority():
    images = {p: np.zeros((80, 60, 3), np.uint8) for p in serve.PHASES}
    tries = hand_tries("at shoulder height or below", "above the head",
                       "at shoulder height or below")  # fmt: skip
    vlm = SeedVLM(check("yes"), check("no"), *tries,
                  *(check(a) for _ in range(4) for a in ("yes", "no")),
                  {"observation": "Sostén el brazo."})  # fmt: skip
    result = base.feedback(serve, images, vlm, "full", "v19", seed=5)
    check_ = result["checks"]["toss_arm_drops_early"]
    assert (result["category"], check_["answer"]) == ("toss_arm_drops_early", "no")
    assert check_["votes"] == {"question 1": "no", "question 2": "yes", "question 3": "no"}
    assert vlm.seeds[2:8] == [5, 5, 6, 6, 7, 7]  # box and question of each try


def test_v19_answers_unclear_without_a_majority():
    images = {p: np.zeros((80, 60, 3), np.uint8) for p in serve.PHASES}
    tries = hand_tries("at shoulder height or below", "above the head", "unclear")
    vlm = FakeVLM(check("yes"), check("no"), *tries,
                  *(check(a) for _ in range(4) for a in ("yes", "no")))  # fmt: skip
    result = base.feedback(serve, images, vlm, "full", "v19")
    assert result["checks"]["toss_arm_drops_early"]["answer"] == "unclear"
    assert result["category"] == "no_clear_problem" and not vlm.answers
