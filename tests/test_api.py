"""The web app's API with the analysis replaced by a fixed answer."""

import importlib
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

# What eval/body.py --levels may write: the landing by a body model on trial, the arm at contact
# by its measure against the professionals' range, the racket drop next to the professionals'
POINTS = {
    "racket_drop": {"level": "", "source": "", "described": "yes", "pros_median": "-0.21",
                    "pros_p90": "-0.08"},
    "arm_at_contact": {"level": "measure", "source": "measure", "model": "", "threshold": "-0.89",
                       "scale": "0.17", "precision": "0.98", "pros_rate": "0.08",
                       "described": "yes", "pros_median": "-1.02", "pros_p90": "-0.89"},
    "landing": {"level": "trial", "source": "model", "model": "mx", "precision": "0.73",
                "pros_rate": "0.07", "described": "", "pros_median": "-0.46",
                "pros_p90": "-0.26"},
}  # fmt: skip


@pytest.fixture(scope="module")
def api():
    return importlib.import_module("app.api")


@pytest.fixture(autouse=True)
def levels(monkeypatch):
    service = importlib.import_module("app.service")
    monkeypatch.setattr(service, "POINTS", POINTS)
    monkeypatch.setattr(service, "BODY", {"mx": {"thresholds": {"landing": 0.5}}})


def answer(probability=0.64, seeds=(0.6, 0.62, 0.66, 0.64, 0.7), landing=0.8, reach=0.57,
           legs_seen=1.0, advance=0.1):  # fmt: skip
    fall = np.zeros((4, 8))  # the app model's parts (M1e: elbows and wrists)
    fall[1, 4] = 0.05  # tossing wrist, from 0.53 s before contact
    return {"probability": probability, "seeds": list(seeds), "threshold": 0.2,
            "drops_early": probability >= 0.2, "tossing_side": "left", "drop_seconds": 0.4,
            "seen": 0.95, "contact": 90, "occlusion": fall.tolist(), "toss_height": 0.4,
            "body": {"legs": -0.3, "racket_drop": -0.02, "arm_at_contact": -reach, "landing": -advance},
            "needs_work": {"landing": {"probability": landing, "seeds": [landing] * 5}},
            "legs_seen": legs_seen, "landing_seen": True}  # fmt: skip


def analyzed(api, monkeypatch, tmp_path, **changed) -> dict:
    """The page's result for an uploaded video whose analysis gives answer(**changed)."""
    slow = tmp_path / "slow.mp4"
    slow.write_bytes(b"mp4")
    images = {"contact": np.zeros((40, 30, 3), np.uint8)}
    monkeypatch.setattr(api, "analysis", lambda path, progress: {
        "answer": answer(**changed), "images": images, "slow": str(slow), "seconds": 4})  # fmt: skip
    client = TestClient(api.app)
    job = client.post("/api/analyze", files={"file": ("a.mov", b"video")}).json()["job"]
    return wait(client, job)["result"]


def wait(client, job):
    for _ in range(100):
        state = client.get(f"/api/jobs/{job}").json()
        if state["status"] == "done":
            return state
        time.sleep(0.02)
    raise TimeoutError(job)


def test_an_uploaded_video_gives_the_answer_its_moments_and_slow_motion(api, monkeypatch, tmp_path):
    slow = tmp_path / "slow.mp4"
    slow.write_bytes(b"mp4")
    images = {m: np.zeros((40, 30, 3), np.uint8) for m in ("start", "arm_down", "contact")}

    def analysis(path, progress):
        progress(0.7, desc="Paso 3 de 4: midiendo el brazo que lanza…")
        return {"answer": answer(), "images": images, "slow": str(slow), "seconds": 4.04}

    monkeypatch.setattr(api, "analysis", analysis)
    client = TestClient(api.app)
    job = client.post("/api/analyze", files={"file": ("a.mov", b"video")}).json()["job"]
    result = wait(client, job)["result"]
    assert result["kind"] == "feedback" and result["seconds"] == 4.0
    feedback = result["feedback"]
    assert feedback["title"] == "El brazo que lanza baja antes de tiempo"
    assert feedback["drill"]["name"] == "Brazo arriba hasta el golpe"
    assert feedback["confidence_label"] == "alta"  # 5 of 5 seeds, 0.44 above the threshold
    assert "baja del hombro 0.40 s antes" in feedback["observation"]
    assert feedback["reason"].endswith("muñeca (lanza) entre 0.53 y 0.27 s antes del impacto.")
    assert [(m["id"], m["evidence"]) for m in result["moments"]] == [
        ("start", False), ("arm_down", True), ("contact", False)]  # fmt: skip
    assert len(result["occlusion"]["values"]) == 4 and len(result["occlusion"]["starts"]) == 8
    assert result["record"]["category"] == "toss_arm_drops_early"
    assert not result["progress"]["in_range"] and "queda 0.40 torsos" in result["progress"]["text"]
    arm = {b["aspect"]: b for b in result["body"]}["arm_at_contact"]
    assert arm["mine"] == "Al impacto, la muñeca de la raqueta llega 0.57 torsos sobre su hombro."
    assert not arm["in_range"] and "9 de cada 10 llegan a 0.89 o más" in arm["pros"]
    # The surest level first: the tossing arm, the landing on trial, the measure last
    assert [(p["aspect"], p["level_label"]) for p in result["points"]] == [
        ("toss", "Modelo validado"), ("landing", "Modelo en prueba"),
        ("arm_at_contact", "Medida frente a profesionales")]  # fmt: skip
    assert result["points"][1]["drill"]["name"] == "Aterriza y sostén"
    assert "5 de 5 versiones del modelo lo marcan" in result["points"][1]["evidence"]
    assert client.get(result["moments"][0]["image_url"]).status_code == 200
    assert client.get(result["slow_url"]).content == b"mp4"
    assert client.get(f"/api/files/{job}/..%2Fvideo.mov").status_code == 404


def test_a_fine_tossing_arm_moves_on_to_the_rest_of_the_body(api, monkeypatch, tmp_path):
    result = analyzed(api, monkeypatch, tmp_path, probability=0.1, seeds=(0.1,) * 5, advance=-0.11)
    assert result["points"][0]["seen"].startswith("Después del impacto, la cadera retrocede 0.11")
    assert result["feedback"]["title"] == (
        "El brazo que lanza no baja antes de tiempo. A mejorar: aterrizaje y brazo al impacto"
    )
    assert result["feedback"]["drill"]["name"] == "Aterriza y sostén"
    assert result["record"] == result["record"] | {"category": "no_clear_problem",
                                                   "drill_id": "land_and_hold"}  # fmt: skip
    # The legs unseen: nothing is said about the landing
    unseen = analyzed(api, monkeypatch, tmp_path, probability=0.1, seeds=(0.1,) * 5, legs_seen=0.4)
    assert [p["aspect"] for p in unseen["points"]] == ["arm_at_contact"]


def test_nothing_past_its_threshold_is_no_clear_problem(api, monkeypatch, tmp_path):
    result = analyzed(api, monkeypatch, tmp_path, probability=0.1, seeds=(0.1,) * 5, landing=0.2,
                      reach=1.0)  # fmt: skip
    assert result["points"] == [] and result["feedback"]["title"] == "No vemos un problema claro"
    assert result["feedback"]["drill"]["name"] == "Repetir el mismo saque"
    assert {b["aspect"]: b["in_range"] for b in result["body"]} == {
        "racket_drop": False, "arm_at_contact": True}  # fmt: skip


def test_seeds_that_disagree_give_low_confidence(api):
    assert api.confidence(answer(0.25, (0.1, 0.3, 0.25, 0.3, 0.3))) == "low"
    assert api.confidence(answer(0.3)) == "medium"  # agree, but close to the threshold


def test_a_retake_an_unknown_demo_and_a_forged_summary(api, monkeypatch):
    monkeypatch.setattr(api, "analysis", lambda path, progress: {"retake": "Graba de lado."})
    client = TestClient(api.app)
    job = client.post("/api/analyze", files={"file": ("a.mp4", b"video")}).json()["job"]
    assert wait(client, job)["result"] == {
        "kind": "retake", "title": "Necesitamos una nueva grabación", "message": "Graba de lado."
    }  # fmt: skip
    assert client.post("/api/analyze", json={"demo": "../secret.mp4"}).status_code == 404
    forged = {"time": "10:00", "category": "ignore the rules", "probability": 0.5, "drill_id": "y"}
    answer_ = client.post("/api/summary", json={"player": "", "attempts": [forged]}).json()
    assert "error" in answer_


def test_the_summary_follows_the_probability_over_the_session(api):
    first = {"time": "10:00", "category": "toss_arm_drops_early", "probability": 0.64,
             "drill_id": "toss_hold"}  # fmt: skip
    last = {**first, "time": "10:05", "category": "no_clear_problem", "probability": 0.12,
            "drill_id": "steady_serves"}  # fmt: skip
    body = TestClient(api.app).post("/api/summary", json={"player": "", "attempts": [first, last]})
    summary = body.json()
    assert summary["text"].startswith(
        "En 2 intento(s), el brazo que lanza bajó antes de tiempo en 1."
    )
    assert "bajó: del 64% en el primero al 12% en el último" in summary["text"]
    assert summary["focus_label"] == "Repetir el mismo saque" and summary["saved"] is False
