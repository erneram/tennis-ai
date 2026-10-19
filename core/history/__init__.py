"""Player sessions, progress summaries, and practice plans.

History is one JSON file per player in data/history/ (ignored by Git: personal data).
"""

import json
import re
from datetime import datetime
from string import Template

from core import ROOT

HISTORY_DIR = ROOT / "data" / "history"
PLAYER_ID = re.compile(r"[A-Za-z0-9_-]{1,40}")
MAX_ATTEMPTS_IN_PROMPT = 30  # bounds the summary prompt to about 2k tokens


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


def attempt(feedback: dict, measurements: dict[str, float]) -> dict:
    return {
        "time": f"{datetime.now():%H:%M}",
        "phase": feedback["phase"],
        "category": feedback["category"],
        "drill_id": feedback["drill_id"],
        "measurements": measurements,
    }


def summarize(vlm, prompts: dict, labels: dict[str, str], sessions: list[dict],
              **generation) -> dict:  # fmt: skip
    """Progress summary and next practice focus.

    `prompts` holds the stroke's summary prompt; `labels` maps each phase, in order, to its name.
    """
    lines = [
        f"- {session['date']}: {a['phase']} / {a['category']}; "
        + ", ".join(f"{name}={value}" for name, value in a["measurements"].items())
        for session in sessions
        for a in session["attempts"]
    ][-MAX_ATTEMPTS_IN_PROMPT:]
    if not lines:
        raise ValueError("No attempts to summarize")
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "maxLength": 600},
            "next_focus": {"type": "string", "maxLength": 300},
            "focus_phase": {"type": "string", "enum": list(labels)},
        },
        "required": ["summary", "next_focus", "focus_phase"],
        "additionalProperties": False,
    }
    phases = ", ".join(f"{phase} ({label})" for phase, label in labels.items())
    user = Template(prompts["user"]).substitute(phases=phases, history="\n".join(lines))
    return vlm.generate_json(prompts["system"], user, [], schema, **generation).data
