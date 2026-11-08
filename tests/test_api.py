"""The web app's API with the analysis replaced by a fixed answer."""

import importlib
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api():
    return importlib.import_module("app.api")


def answer(probability=0.64, seeds=(0.6, 0.62, 0.66, 0.64, 0.7)):
    fall = np.zeros((4, 8))  # the app model's parts (M1e: elbows and wrists)
    fall[1, 4] = 0.05  # tossing wrist, from 0.53 s before contact
    return {"probability": probability, "seeds": list(seeds), "threshold": 0.2,
            "drops_early": probability >= 0.2, "tossing_side": "left", "drop_seconds": 0.4,
            "seen": 0.95, "contact": 90, "occlusion": fall.tolist()}  # fmt: skip


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
    assert client.get(result["moments"][0]["image_url"]).status_code == 200
    assert client.get(result["slow_url"]).content == b"mp4"
    assert client.get(f"/api/files/{job}/..%2Fvideo.mov").status_code == 404


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
