import pytest

from core import history


def test_sessions_accumulate_per_player(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_DIR", tmp_path)
    feedback = {"phase": "trophy", "category": "toss_arm_drops_early", "drill_id": "toss_hold"}
    first = history.attempt(feedback)
    history.save_session("p01", [first])
    sessions = history.save_session("p01", [first, first])
    assert [len(s["attempts"]) for s in sessions] == [1, 2]
    assert history.load("p01") == sessions
    assert history.load("p02") == []


@pytest.mark.parametrize("player", ["../etc/passwd", "a/b", "", "x" * 41, "nombre con espacios"])
def test_player_id_cannot_escape_history_folder(player):
    with pytest.raises(ValueError):
        history.load(player)
