import numpy as np
import pytest

from core.analysis import InsufficientEvidence
from core.metrics import INDEX
from strokes import base, serve

FPS = 30


def synthetic_serve(frames: int = 60) -> np.ndarray:
    """Right-handed serve seen from the side: toss peaks at frame 22, contact at frame 35."""
    standing = {
        "nose": (100, 50), "left_eye": (98, 48), "right_eye": (102, 48),
        "left_ear": (96, 50), "right_ear": (104, 50),
        "left_shoulder": (90, 80), "right_shoulder": (110, 80),
        "left_elbow": (85, 110), "right_elbow": (115, 110),
        "left_wrist": (85, 140), "right_wrist": (115, 140),
        "left_hip": (95, 150), "right_hip": (105, 150),
        "left_knee": (95, 200), "right_knee": (105, 200),
        "left_ankle": (95, 250), "right_ankle": (105, 250),
    }  # fmt: skip
    xy = np.zeros((frames, 17, 2))
    for name, position in standing.items():
        xy[:, INDEX[name]] = position
    t = np.arange(frames)
    # Toss hand: dips at frame 10, rises to its peak at 22, then comes down.
    xy[:, INDEX["left_wrist"], 1] = np.interp(t, [0, 10, 22, 40, 59], [140, 160, 20, 140, 140])
    xy[:, INDEX["left_elbow"], 1] = np.interp(t, [0, 10, 22, 40, 59], [110, 120, 50, 110, 110])
    # Racket hand: rises from frame 25 to its peak above the head at 35.
    xy[:, INDEX["right_wrist"], 1] = np.interp(t, [0, 25, 35, 50, 59], [140, 140, -40, 200, 200])
    xy[:, INDEX["right_elbow"], 1] = np.interp(t, [0, 25, 35, 50, 59], [110, 110, 20, 150, 150])
    xy[:, INDEX["right_wrist"], 0] = xy[:, INDEX["right_elbow"], 0] = 110
    # Knees bend most at frame 30 (knees move forward).
    for side in ("left_knee", "right_knee"):
        xy[:, INDEX[side], 0] += np.interp(t, [0, 22, 30, 35, 59], [0, 10, 25, 0, 0])
    return xy


def test_segment_finds_phases_in_order():
    xy = synthetic_serve()
    phases = serve.segment(xy, FPS)
    assert serve.phases.racket_side(xy) == "right"
    assert phases == {"stance": 0, "toss": 10, "trophy": 22, "leg_drive": 30,
                      "contact": 35, "follow_through": 36}  # fmt: skip
    frames = serve.key_frames(phases, len(xy), FPS)
    assert frames["stance"] == 4 and frames["follow_through"] == 47


def test_measure_and_rule_feedback():
    xy = synthetic_serve()
    values = serve.measure(xy, FPS, serve.segment(xy, FPS))
    assert values["contact_arm_extension_deg"] == pytest.approx(180)
    assert values["contact_height"] > 1
    assert values["knee_bend_deg"] > 20

    feedback = base.rule_feedback(serve, values)
    drills = {d["id"] for d in base.load(serve, "drills.yaml")["drills"]}
    assert feedback["phase"] in serve.PHASES and feedback["drill_id"] in drills


def test_every_category_has_a_drill_and_a_known_phase():
    criteria = base.load(serve, "criteria.yaml")
    drill_categories = {d["category"] for d in base.load(serve, "drills.yaml")["drills"]}
    assert set(criteria["categories"]) == drill_categories
    assert {c["phase"] for c in criteria["categories"].values()} <= set(serve.PHASES)
    assert set(criteria["phases"]) == set(serve.PHASES)


def test_clip_ending_at_contact_asks_for_new_recording():
    with pytest.raises(InsufficientEvidence, match="termina en el impacto"):
        serve.segment(synthetic_serve()[:38], FPS)
