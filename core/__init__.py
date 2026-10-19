"""Shared core used by every analysis. Must not contain stroke-specific logic (see tests/test_architecture.py)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
