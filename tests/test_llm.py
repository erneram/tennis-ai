from types import SimpleNamespace

import numpy as np
import pytest

from core.llm import Generation, validate
from strokes import base, serve


class FakeVLM:
    """Records the prompt and answers like the real model would."""

    def generate_json(self, system, user, images, schema, **generation):
        self.user, self.images = user, images
        data = {"phase": "trophy", "category": "racket_elbow_low", "observation": "Sube el codo.",
                "evidence_phases": ["trophy"], "drill_id": "trophy_pause", "confidence": "high"}  # fmt: skip
        validate(data, schema)
        return Generation(
            data, prompt_tokens=900, generation_tokens=60, seconds=2.0, peak_memory_gb=4
        )


def analysis(measurements):
    image = np.zeros((8, 8, 3), np.uint8)
    return SimpleNamespace(measurements=measurements, images=dict.fromkeys(serve.PHASES, image))


def test_full_prompt_marks_out_of_range_measurements():
    vlm = FakeVLM()
    result = base.feedback(
        serve, analysis({"racket_elbow_height": -0.4, "toss_to_contact_s": 1.1}), vlm
    )
    assert "racket_elbow_height = -0.4 (coach range: min -0.15) -> OUT OF RANGE" in vlm.user
    assert "toss_to_contact_s = 1.1 (informative)" in vlm.user
    assert "$" not in vlm.user and len(vlm.images) == len(serve.PHASES)
    assert result["consistent"] and result["source"] == "vlm:full:v1"


def test_simple_baseline_prompt_has_no_measurements():
    vlm = FakeVLM()
    base.feedback(serve, analysis({"racket_elbow_height": -0.4}), vlm, variant="simple")
    assert "racket_elbow_height" not in vlm.user and "trophy_pause" in vlm.user


def test_validate_rejects_unknown_values():
    schema = base.feedback_schema(serve)
    good = {"phase": "contact", "category": "low_contact_point", "observation": "x",
            "evidence_phases": ["contact"], "drill_id": "higher_toss", "confidence": "low"}  # fmt: skip
    validate(good, schema)
    with pytest.raises(ValueError):
        validate({**good, "drill_id": "invented"}, schema)
    with pytest.raises(ValueError):
        validate({**good, "evidence_phases": ["warmup"]}, schema)
