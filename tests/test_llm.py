from types import SimpleNamespace

import numpy as np
import pytest

from core.llm import Generation, validate
from strokes import base, serve


class FakeVLM:
    """Records the prompt and answers like the real model would."""

    def generate_json(self, system, user, images, schema, **generation):
        self.system, self.user, self.images = system, user, images
        data = {"phase": "toss", "category": "toss_arm_bent", "observation": "Estira el brazo.",
                "evidence_phases": ["toss"], "drill_id": "toss_target", "confidence": "high"}  # fmt: skip
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
        serve, analysis({"toss_arm_extension_deg": 120, "toss_to_contact_s": 1.1}), vlm
    )
    assert "toss_arm_extension_deg = 120 (reference range: min 150) -> OUT OF RANGE" in vlm.user
    assert "toss_to_contact_s = 1.1 (informative)" in vlm.user
    assert "$" not in vlm.user and len(vlm.images) == len(serve.PHASES)
    assert result["consistent"] and result["source"] == "vlm:full:v1"


def test_simple_baseline_prompt_has_no_measurements():
    vlm = FakeVLM()
    base.feedback(serve, analysis({"toss_arm_extension_deg": 120}), vlm, variant="simple")
    assert "toss_arm_extension_deg" not in vlm.user and "toss_target" in vlm.user


def test_validate_rejects_unknown_values():
    schema = base.feedback_schema(serve)
    good = {"phase": "contact", "category": "low_contact_point", "observation": "x",
            "evidence_phases": ["contact"], "drill_id": "higher_toss", "confidence": "low"}  # fmt: skip
    validate(good, schema)
    with pytest.raises(ValueError):
        validate({**good, "drill_id": "invented"}, schema)
    with pytest.raises(ValueError):
        validate({**good, "evidence_phases": ["warmup"]}, schema)


def test_evidence_limits_the_categories_the_model_can_choose():
    criteria = base.load(serve, "criteria.yaml")
    out_of_range = {"toss_arm_extension_deg": 120, "toss_to_contact_s": 1.5}
    assert base.evidence_categories(criteria, out_of_range) == ["toss_arm_bent"]

    allowed = base.evidence_categories(criteria, {**out_of_range, "toss_arm_extension_deg": 175})
    assert "toss_arm_bent" not in allowed
    assert "low_contact_point" in allowed  # not measurable: left to the frames

    schema = base.feedback_schema(serve, ["toss_arm_bent"])
    assert schema["properties"]["phase"]["enum"] == ["toss"]
    assert schema["properties"]["drill_id"]["enum"] == ["toss_target"]


def test_v3_prompt_sends_the_restricted_schema():
    vlm = FakeVLM()
    result = base.feedback(serve, analysis({"toss_arm_extension_deg": 120}), vlm, version="v3")
    assert result["category"] == "toss_arm_bent" and result["source"] == "vlm:full:v3"
    assert "practices alone" in vlm.system
