# Serve Lab

Interactive assistant that analyzes the **tennis serve** from a short video. It splits the motion into phases, finds the phase with the main opportunity for improvement, shows the frames that support the observation, and suggests a drill. It keeps a history of sessions to follow the player's progress.

Project 3 of CC3092 Deep Learning y Sistemas Inteligentes (UVG). Full proposal: [docs/proposal.md](docs/proposal.md).

## Requirements

- macOS on Apple Silicon (developed on a Mac M5 Pro, 24 GB).
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 automatically if needed.

## Installation

```bash
uv sync                  # default install: MediaPipe pose backend
uv sync --group yolo     # optional: YOLO Pose backend (AGPL-3.0, see docs/licenses.md)
cp .env.example .env     # optional settings
```

## Usage

*To be completed in Phase 4 (interface in `app/`).*

## Evaluation

*To be completed in Phase 3: how to run `eval/run_eval.py` and reproduce the reported metrics.*

## Tests

```bash
uv run pytest
```

## Repository layout

```
configs/      shared settings (pose backend, model, generation parameters)
core/         code shared by all analyses: video, pose, metrics, llm, history
strokes/      one module per stroke; serve/ is the current prototype
app/          user interface
data/         videos (not in Git) and annotations, see data/README.md
eval/         evaluation sets and metric scripts, per stroke
experiments/  iteration log (iterations.csv)
docs/         proposal, licenses, images
tests/        automated checks
```

Rules that keep the repository scalable:

- `core/` contains no stroke-specific logic. `tests/test_architecture.py` fails if `core/` imports `strokes`.
- Each stroke lives in its own folder under `strokes/` with the same structure.
- Criteria, drills, and prompts are versioned files, not hard-coded strings.
- Each stroke has its own evaluation set under `eval/`.

## Licenses

Third-party models, libraries, and datasets: [docs/licenses.md](docs/licenses.md). The project code has no license yet (all rights reserved).
