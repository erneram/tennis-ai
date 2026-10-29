# Serve Lab

Interactive assistant that analyzes the **tennis serve** from a short video. It splits the motion into phases, finds the phase with the main opportunity for improvement, shows the frames that support the observation, and suggests a drill. It keeps a history of sessions to follow the player's progress.

Project 3 of CC3092 Deep Learning y Sistemas Inteligentes (UVG). Only a vision-language model looks at the video: there is no pose model and there are no measurements. Full proposal: [docs/proposal.md](docs/proposal.md). The Final Project, which adds pose estimation, lives on the `proyecto-final` branch.

## Requirements

- macOS on Apple Silicon (developed on a Mac M5 Pro, 24 GB).
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 automatically if needed.

## Installation

```bash
uv sync                  # install
cp .env.example .env     # optional settings
```

The first run downloads Qwen3-VL-4B (about 3 GB) into the Hugging Face cache. After that, everything runs offline on the Mac.

## Usage

```bash
uv run python -m app     # web app at http://127.0.0.1:7860
```

How one video is analyzed (prompt v6):

1. OpenCV checks the video (frame rate, length) and takes 32 evenly spaced frames.
2. The model decides whether the video can be analyzed and where the serve starts and ends. Players bounce the ball before serving, so most of a video is not the serve.
3. OpenCV takes 32 frames again, only inside the serve, and the model picks the frame that best shows each of the six phases. Each frame has its number drawn on it.
4. One yes / no / unclear question per category, each about the key frame of its phase only (`check` in `strokes/serve/criteria.yaml`). The earliest phase with a clear "no" is the point to improve, with its drill; with none, the answer is "no clear problem". One more call writes the observation.

Every answer is JSON constrained by a schema, so the model can only choose known frames, phases, categories, and drills. Why each step exists is in the header of each prompt version, `strokes/serve/prompts/v*.yaml`.

In the app, upload one serve recorded from the side, following [docs/grabacion.md](docs/grabacion.md) (recording guide, in Spanish). The app shows:

- the video in slow motion with the current phase,
- the phase with the main opportunity for improvement, the frames that support it (★), and a drill,
- a comparison with the previous attempt.

"Terminar sesión" summarizes the session and proposes the focus of the next practice. With a player ID, the session is added to that player's history in `data/history/`.

If the video cannot be analyzed (wrong angle, player out of frame, too dark, too short), the app asks for a new recording instead of giving feedback. If the model gives no valid answer, the app says so and asks to try again.

### Demo mode (fair)

One analysis takes 15 to 45 s on the Mac. For the fair, put the demo videos in `data/demo/` (only players with `consent_image` = `yes`; include one bad recording to show the request for a new one) and run:

```bash
uv run python -m app --demo
```

At start, each video in `data/demo/` is analyzed once; the answer is saved in `data/processed/demo_cache/` and shown at once after that, marked "respuesta guardada". The videos appear as examples under the upload box. Any other video, such as a visitor's, is analyzed live and its answer is not saved. A saved answer is reused only for the same video, model, prompt version and criteria version; delete `data/processed/demo_cache/` to analyze again.

## Evaluation

The labels written by the team are only the answer key: in Project 3 the model is not trained, it is used as downloaded with a prompt. See [what the labels are for](data/README.md#what-the-labels-are-for).

1. Put the videos in `data/raw/`. There is no coach: both members label each serve on their own, compare, and write the agreed labels in `eval/serve/labels.csv` (guide in [data/README.md](data/README.md)):

```bash
uv run python -m eval.run_eval --agreement eval/serve/labels_ernesto.csv eval/serve/labels_hugo.csv
```

2. Run the full system and the two baselines:

```bash
uv run python -m eval.run_eval --variant full   --note "v4 prompt"
uv run python -m eval.run_eval --variant simple --note "baseline: frames only, no criteria"
uv run python -m eval.run_eval --model mlx-community/Qwen3-VL-2B-Instruct-4bit --note "baseline: smaller model"
```

Each run saves its predictions in `eval/serve/runs/` and appends a row to `experiments/iterations.csv`: phase accuracy, category F1, hallucination rate, correct re-recording requests, latency, tokens (both calls), and peak memory. The hallucination rate in the file is an automatic proxy (the category belongs to another phase); both members also read each observation next to its frames, and that review goes in the written report.

Settings that change results (number of frames, language model, prompt version, temperature) are in `configs/default.yaml`. Change one at a time and run the evaluation again.

## Privacy

- Videos and player histories stay in ignored folders under `data/`. Nothing is uploaded: the app listens only on `127.0.0.1` and Gradio analytics are disabled.

## Tests

```bash
uv run pytest
```

## Repository layout

```
configs/      shared settings (frames, model, generation parameters)
core/         code shared by all analyses: video, frames, llm, history
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
