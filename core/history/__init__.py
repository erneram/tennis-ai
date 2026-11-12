"""Player sessions: the attempts of each practice, to follow a player's progress.

History is one JSON file per player in data/history/ (ignored by Git: personal data).
"""

import json
import re
from datetime import datetime

from core import ROOT

HISTORY_DIR = ROOT / "data" / "history"
PLAYER_ID = re.compile(r"[A-Za-z0-9_-]{1,40}")


def _file(player: str):
    if not PLAYER_ID.fullmatch(player):  # the ID becomes a file name: no paths allowed
        raise ValueError("El ID de jugador solo puede tener letras, números, - y _ (máximo 40).")
    return HISTORY_DIR / f"{player}.json"


def load(player: str) -> list[dict]:
    file = _file(player)
    return json.loads(file.read_text()) if file.exists() else []


def save_session(player: str, attempts: list[dict]) -> list[dict]:
    """Append one session and return the player's full history."""
    file = _file(player)
    sessions = load(player) + [{"date": f"{datetime.now():%Y-%m-%d %H:%M}", "attempts": attempts}]
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    partial = file.with_suffix(".part")
    partial.write_text(json.dumps(sessions, ensure_ascii=False, indent=1))
    partial.replace(file)  # atomic: a crash never leaves a half-written history
    return sessions


def attempt(category: str, probability: float, drill_id: str, points: list[str]) -> dict:
    """One analyzed serve of a session: the model's answer for the tossing arm, the points to
    improve it got and the drill of the first."""
    return {
        "time": f"{datetime.now():%H:%M}",
        "category": category,
        "probability": round(probability, 3),
        "drill_id": drill_id,
        "points": points,
    }
