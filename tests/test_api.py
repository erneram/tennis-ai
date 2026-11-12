"""The web app's API with the analysis replaced by a fixed answer."""

import importlib
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

# What eval/body.py --levels may write: the tossing arm by the app's model, the legs by an
# exploratory body model, the landing by one on trial, the arm at contact by its measure against
# the professionals' range, and the racket drop next to the professionals' with no point
POINTS = {
    "toss": {"level": "value", "source": "model", "model": "m1e", "precision": "0.2",
             "pros_rate": "0.0", "described": ""},
    "legs": {"level": "exploratory", "source": "model", "model": "mx", "precision": "0.7",
             "pros_rate": "0.0", "new_player": "0.585", "described": "", "pros_median": "-0.30",
             "pros_p90": "-0.11"},
    "racket_drop": {"level": "", "source": "", "described": "yes", "precision": "0.49",
                    "pros_median": "-0.21", "pros_p90": "-0.08"},
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
    monkeypatch.setattr(service, "BODY", {"mx": {"thresholds": {"landing": 0.5, "legs": 0.5}}})


def answer(probability=0.64, seeds=(0.6, 0.62, 0.66, 0.64, 0.7), landing=0.8, reach=0.57,
           legs_seen=1.0, advance=0.1, legs=0.2):  # fmt: skip
    fall = np.zeros((4, 8))  # the app model's parts (M1e: elbows and wrists)
    fall[1, 4] = 0.05  # tossing wrist, from 0.53 s before contact
    return {"probability": probability, "seeds": list(seeds), "threshold": 0.2,
            "drops_early": probability >= 0.2, "tossing_side": "left", "drop_seconds": 0.4,
            "seen": 0.95, "contact": 90, "occlusion": fall.tolist(), "toss_height": 0.4,
            "body": {"legs": -0.3, "racket_drop": -0.02, "arm_at_contact": -reach, "landing": -advance},
            "needs_work": {"landing": {"probability": landing, "seeds": [landing] * 5},
                           "legs": {"probability": legs, "seeds": [legs] * 5}},
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


def test_an_uploaded_video_gives_the_points_its_moments_and_slow_motion(api, monkeypatch, tmp_path):
    slow = tmp_path / "slow.mp4"
    slow.write_bytes(b"mp4")
    images = {m: np.zeros((40, 30, 3), np.uint8) for m in ("bend", "arm_down", "contact")}

    def analysis(path, progress):
        progress(0.7, desc="Paso 3 de 4: midiendo tu saque…")
        return {"answer": answer(), "images": images, "slow": str(slow), "seconds": 4.04}

    monkeypatch.setattr(api, "analysis", analysis)
    client = TestClient(api.app)
    job = client.post("/api/analyze", files={"file": ("a.mov", b"video")}).json()["job"]
    result = wait(client, job)["result"]
    assert result["kind"] == "feedback" and result["seconds"] == 4.0
    assert result["title"] == "A mejorar: brazo al impacto, aterrizaje y brazo que lanza"
    # The point that was right most often on validation first, whatever its level
    assert [(p["aspect"], p["level_label"], p["precision"]) for p in result["points"]] == [
        ("arm_at_contact", "Medida frente a profesionales", 0.98),
        ("landing", "Modelo en prueba", 0.73), ("toss", "Modelo elegido por valor", 0.2)]  # fmt: skip
    arm, landing, toss = result["points"]
    assert arm["drill"]["name"] == "Alcanzar el punto más alto"
    # What the player reads has no numbers: what the app saw, what to aim for, how far to trust it
    assert arm["seen"].startswith("Le pegas a la pelota más abajo de lo que alcanzas")
    assert arm["goal"].startswith("Lo ideal: el brazo estirado")
    assert landing["seen"] == "Después de pegarle avanzas poco: caes casi en el mismo lugar."
    assert toss["seen"].startswith("Bajas muy pronto el brazo con el que lanzas la pelota")
    assert [p["often"] for p in result["points"]] == [
        "casi siempre", "7 de cada 10 veces", "2 de cada 10 veces"]  # fmt: skip
    assert toss["trust"] == "Tómalo como una sugerencia, no como un hecho."
    assert not any(
        c.isdigit() for p in result["points"] for c in p["seen"] + p["goal"] + p["trust"]
    )
    # The numbers are in the technical detail of each point
    assert arm["detail"].startswith("Medida frente a profesionales. Al impacto, la muñeca de la "
                                    "raqueta llega 0.57 torsos")  # fmt: skip
    assert "5 de 5 versiones del modelo lo marcan" in landing["detail"]
    assert "baja del hombro 0.40 s antes" in toss["detail"]
    assert "En validación acertó 20% de las veces que lo marcó" in toss["detail"]
    # The other parts the app looked at, in the order of the motion
    assert [(o["aspect"], o["status"]) for o in result["others"]] == [
        ("legs", "fine"), ("racket_drop", "outside")]  # fmt: skip
    legs, drop = result["others"]
    assert legs["seen"] == "Doblas las rodillas antes de subir a pegarle." and "goal" not in legs
    assert drop["seen"].startswith("Antes de subir a pegarle, tu raqueta casi no cae")
    assert drop["goal"].startswith("Lo ideal") and "acertó 5 de cada 10 veces" in drop["note"]
    assert [(m["id"], m["evidence"]) for m in result["moments"]] == [
        ("bend", False), ("arm_down", True), ("contact", True)]  # fmt: skip
    assert result["moments"][2]["note"] == "En naranja: el brazo de la raqueta."
    model = result["toss"]
    assert model["confidence_label"] == "alta"  # 5 of 5 seeds, 0.44 above the threshold
    assert model["reason"].endswith("muñeca (lanza) entre 0.53 y 0.27 s antes del impacto.")
    assert len(model["occlusion"]["values"]) == 4 and len(model["occlusion"]["starts"]) == 8
    assert result["record"] == result["record"] | {
        "category": "toss_arm_drops_early", "drill_id": "high_reach",
        "points": ["arm_at_contact", "landing", "toss"]}  # fmt: skip
    assert [c["aspect"] for c in result["compare"]] == [
        "toss_height", "racket_drop", "arm_at_contact"]  # fmt: skip
    height, _, reach = result["compare"]
    assert not height["in_range"] and height["value"] == 0.4
    assert reach["caption"] == "Qué tan arriba le pegas a la pelota."
    assert (reach["value"], reach["pros_median"], reach["pros_floor"]) == (0.57, 1.02, 0.89)
    assert not reach["in_range"]
    assert client.get(result["moments"][0]["image_url"]).status_code == 200
    assert client.get(result["slow_url"]).content == b"mp4"
    assert client.get(f"/api/files/{job}/..%2Fvideo.mov").status_code == 404


def test_of_four_points_past_their_threshold_the_three_surest_are_given(api, monkeypatch, tmp_path):
    result = analyzed(api, monkeypatch, tmp_path, legs=0.9)
    assert [p["aspect"] for p in result["points"]] == ["arm_at_contact", "landing", "legs"]
    assert result["points"][2]["seen"] == "Doblas poco las rodillas antes de subir a pegarle."
    assert result["points"][2]["trust"].startswith("Con un jugador que la app nunca ha visto")
    assert "Es exploratorio" in result["points"][2]["detail"]
    assert result["points"][2]["drill"]["name"] == "Carga de piernas"
    # The tossing arm is still said, among the other parts
    toss = result["others"][0]
    assert (toss["aspect"], toss["status_label"]) == ("toss", "A mejorar")
    assert toss["note"].startswith("También lo marcamos, pero damos 3 puntos como máximo")


def test_a_fine_tossing_arm_is_one_more_part_looked_at(api, monkeypatch, tmp_path):
    result = analyzed(api, monkeypatch, tmp_path, probability=0.1, seeds=(0.1,) * 5, advance=-0.11)
    assert result["title"] == "A mejorar: brazo al impacto y aterrizaje"
    assert result["points"][1]["seen"] == (
        "Después de pegarle te vas hacia atrás en vez de entrar a la cancha."
    )
    assert "la cadera retrocede 0.11 torsos" in result["points"][1]["detail"]
    assert result["drill"]["name"] == "Alcanzar el punto más alto"
    toss = result["others"][0]
    assert (toss["aspect"], toss["status_label"]) == ("toss", "Sin problema")
    assert toss["seen"].startswith("Mantienes arriba el brazo con el que lanzas la pelota")
    assert result["record"] == result["record"] | {"category": "no_clear_problem",
                                                   "drill_id": "high_reach"}  # fmt: skip
    # The legs unseen: nothing is said about them or the landing
    unseen = analyzed(api, monkeypatch, tmp_path, probability=0.1, seeds=(0.1,) * 5, legs_seen=0.4)
    assert [p["aspect"] for p in unseen["points"]] == ["arm_at_contact"]
    assert {o["aspect"]: o["status"] for o in unseen["others"]} == {
        "toss": "fine", "legs": "unseen", "racket_drop": "outside", "landing": "unseen"}  # fmt: skip


def test_hips_that_do_not_go_down_are_said_in_words():
    service = importlib.import_module("app.service")
    still = {"body": {"legs": 0.04}}  # the hips stay above where they stood
    assert service.described("legs", "improve", still) == (
        "Casi no doblas las rodillas antes de subir a pegarle."
    )
    assert service.measured("legs", still)[0] == (
        "Antes del salto, la cadera no baja: queda 0.04 torsos más arriba que al empezar."
    )


def test_nothing_past_its_threshold_is_no_clear_problem(api, monkeypatch, tmp_path):
    result = analyzed(api, monkeypatch, tmp_path, probability=0.1, seeds=(0.1,) * 5, landing=0.2,
                      reach=1.0)  # fmt: skip
    assert result["points"] == [] and result["title"] == "No vemos un problema claro"
    assert result["drill"]["name"] == "Repetir el mismo saque"
    assert {c["aspect"]: c["in_range"] for c in result["compare"]} == {
        "toss_height": False, "racket_drop": False, "arm_at_contact": True}  # fmt: skip
    assert result["record"]["points"] == []


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
    # The racket drop is an aspect with no drill, so the app never gives it as a point
    no_drill = {"time": "10:00", "category": "no_clear_problem", "probability": 0.5,
                "drill_id": "steady_serves", "points": ["racket_drop"]}  # fmt: skip
    answer_ = client.post("/api/summary", json={"player": "", "attempts": [no_drill]}).json()
    assert "error" in answer_


def test_the_summary_counts_each_point_over_the_session(api):
    first = {"time": "10:00", "category": "toss_arm_drops_early", "probability": 0.64,
             "drill_id": "high_reach", "points": ["arm_at_contact", "toss"]}  # fmt: skip
    last = {**first, "time": "10:05", "category": "no_clear_problem", "probability": 0.12,
            "points": ["arm_at_contact"]}  # fmt: skip
    client = TestClient(api.app)
    summary = client.post("/api/summary", json={"player": "", "attempts": [first, last]}).json()
    assert summary["text"] == (
        "En 2 intento(s) salió: brazo al impacto en 2, brazo que lanza en 1."
    )
    assert summary["focus_label"] == "Alcanzar el punto más alto" and summary["saved"] is False
    clear = {**last, "drill_id": "steady_serves", "points": []}
    summary = client.post("/api/summary", json={"player": "", "attempts": [clear]}).json()
    assert summary["text"] == "En 1 intento(s) no vimos un problema claro."
    assert summary["focus_label"] == "Repetir el mismo saque"
