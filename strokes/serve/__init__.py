"""Serve module: phases, measurements, coach criteria, drills, and versioned prompts."""

from strokes.serve.metrics import MEASUREMENT_LABELS, measure
from strokes.serve.phases import LABELS, PHASES, key_frames, segment

NAME = "serve"

__all__ = ["LABELS", "MEASUREMENT_LABELS", "NAME", "PHASES", "key_frames", "measure", "segment"]
