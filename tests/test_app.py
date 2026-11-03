"""The app's answer card and session flow, without loading the model."""

import importlib

import numpy as np
import pytest

import core.llm


@pytest.fixture(scope="module")
def app():
    original, core.llm.VLM = core.llm.VLM, lambda name: None  # the app loads it at import
    try:
        yield importlib.import_module("app.__main__")
    finally:
        core.llm.VLM = original


def arm_feedback(observation="Tu brazo baja pronto."):
    checks = {
        "toss_arm_bent": {"seen": "", "answer": "yes", "phase": "trophy"},
        "toss_arm_drops_early": {"seen": "", "answer": "no", "phase": "leg_drive"},
        "insufficient_knee_bend": {"seen": "", "answer": "unclear", "phase": "leg_drive"},
    }
    return {"phase": "trophy", "category": "toss_arm_drops_early", "observation": observation,
            "evidence_phases": ["leg_drive"], "drill_id": "toss_hold", "confidence": "medium",
            "consistent": True, "checks": checks}  # fmt: skip


def test_the_card_escapes_the_model_text_and_says_where_the_problem_is_seen(app):
    card = app.result_html(arm_feedback("<script>x</script>"), 31.4, 2, cached=True)
    assert "<script>" not in card and "&lt;script&gt;" in card
    assert "Fase a mejorar: 3. Posición de trofeo" in card and "Se ve en 4. Impulso" in card
    assert "Intento 2 · análisis en 31 s · respuesta guardada" in card
    assert "Pudimos revisar 2 de 3 puntos" in card


def test_the_strip_shows_six_phases_with_their_verdict(app):
    strip = app.phase_strip(arm_feedback()["checks"], {"leg_drive"})
    assert strip.count('class="sl-chip') == 6
    assert '<div class="sl-chip none"><b>—</b>1. Posición inicial' in strip
    assert '<div class="sl-chip yes"><b>✅</b>3. Posición de trofeo' in strip
    assert '<div class="sl-chip no"><b>❌ ★</b>4. Impulso de piernas' in strip


def test_the_comparison_shows_from_the_second_attempt(app, monkeypatch):
    images = {p: np.zeros((40, 30, 3), np.uint8) for p in app.STROKE.PHASES}
    result = {"feedback": arm_feedback(), "images": images, "slow": "slow.mp4", "seconds": 30.0}
    monkeypatch.setattr(app, "analysis", lambda path, progress: result)
    first = app.analyze_attempt("serve.mp4", [], progress=None)
    second = app.analyze_attempt("serve.mp4", first[-1], progress=None)
    assert first[6]["visible"] is False and second[6]["visible"] is True
    assert "Intento 2" in second[1] and len(second[-1]) == 2
    assert app.analyze_attempt("serve.mp4", second[-1], progress=None)[2]["visible"] is True


def test_a_retake_and_a_new_player_empty_every_output(app, monkeypatch):
    monkeypatch.setattr(app, "analysis", lambda path, progress: {"retake": "Graba de lado."})
    out = app.analyze_attempt("serve.mp4", [], progress=None)
    assert len(out) == 10 and "Necesitamos una nueva grabación" in out[1]
    assert "Cómo grabar el saque" in out[0] and out[5]["visible"] is False
    assert len(app.start_over()) == 12 and app.start_over()[2] == app.WELCOME
