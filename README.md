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

The first run downloads the models (about 3 GB for Qwen3-VL-4B, plus the pose model) into the Hugging Face cache and `models/`. After that, everything runs offline on the Mac.

## Usage

```bash
uv run python -m app                              # web app at http://127.0.0.1:7860
uv run python -m core.pose data/raw/p01_s01.mp4   # pose overlay only (development)
```

In the app, upload one serve recorded from the side. The app shows:

- the video in slow motion with the pose and the current phase,
- the phase with the main opportunity for improvement, the frames that support it (★), and a drill,
- the measurements against reference ranges based on published studies,
- a comparison with the previous attempt.

"Terminar sesión" summarizes the session and proposes the focus of the next practice. With a player ID, the session is added to that player's history in `data/history/`.

If the video cannot be analyzed (wrong angle, player out of frame, too dark, too short), the app asks for a new recording instead of giving feedback. If the model fails, the app falls back to the rule-based recommendation and says so.

## Evaluation

1. Put the videos in `data/raw/`. There is no coach: both members label each serve on their own, compare, and write the agreed labels in `eval/serve/labels.csv` (guide in [data/README.md](data/README.md)):

```bash
uv run python -m eval.run_eval --agreement eval/serve/labels_ernesto.csv eval/serve/labels_hugo.csv
```

2. Run the full system and the two baselines:

```bash
uv run python -m eval.run_eval --variant full   --note "v3 prompt"
uv run python -m eval.run_eval --variant simple --note "baseline: frames only"
uv run python -m eval.run_eval --variant rules  --note "baseline: no language model"
```

Each run saves its predictions in `eval/serve/runs/` and appends a row to `experiments/iterations.csv`: phase accuracy, category F1, hallucination rate, correct re-recording requests, latency, tokens, and peak memory.

Settings that change results (pose model, language model, prompt version, temperature) are in `configs/default.yaml`. Change one at a time and run the evaluation again.

## Privacy

- Videos, pose caches, and player histories stay in ignored folders under `data/`. Nothing is uploaded: the app listens only on `127.0.0.1` and Gradio analytics are disabled.
- Ultralytics (optional YOLO backend) sends anonymous usage statistics by default. Turn them off once with `uv run yolo settings sync=False`.

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
