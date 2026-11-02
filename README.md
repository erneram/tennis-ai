# Serve Lab

Interactive assistant that analyzes the **tennis serve** from a short video. It splits the motion into phases, finds the phase with the main opportunity for improvement, shows the frames that support the observation, and suggests a drill. It keeps a history of sessions to follow the player's progress.

Project 3 of CC3092 Deep Learning y Sistemas Inteligentes (UVG). Only a vision-language model looks at the video: there is no pose model and there are no measurements. Full proposal: [docs/proposal.md](docs/proposal.md). The Final Project, which adds pose estimation, lives on the `proyecto-final` branch.

## Requirements

- macOS on Apple Silicon (developed on a Mac M5 Pro, 24 GB).
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 automatically if needed.
- [Node.js](https://nodejs.org/) 20 or later, only to build the web page once.

## Installation

```bash
uv sync                  # install
cp .env.example .env     # optional settings
cd web && npm install && npm run build && cd ..   # build the web page into web/dist
```

The first run downloads Qwen3-VL-4B (about 3 GB) into the Hugging Face cache. After that, everything runs offline on the Mac.

## Usage

```bash
uv run python -m app.api     # web app at http://127.0.0.1:7860
```

The page is a React app in `web/` (components from [The Gridcn](https://thegridcn.com), how to build and change it in [web/README.md](web/README.md)), served with its API by `app/api.py`. `uv run python -m app` starts the earlier Gradio version instead, kept as a fallback: it gives the same answers.

How one video is analyzed (prompt v21):

1. OpenCV checks the video (frame rate, length) and takes 32 evenly spaced frames.
2. The model draws a box around the player on the first, middle and last frames. The box is joined, grown to the reach of a serve (toss, racket, landing), and every frame is decoded again cropped to it, so a player far from the camera fills the frame. No crop when it would keep most of the frame.
3. The model decides whether the video can be analyzed and where the serve starts and ends. Players bounce the ball before serving, so most of a video is not the serve.
4. OpenCV takes 32 frames again, only inside the serve, and the model picks the frame that best shows each of the six phases. Each frame has its number drawn on it. When the answer is six frames in a row (the model counted instead of looking), it tries again on every 2nd frame, then on every 3rd. Contact lasts a few hundredths of a second, so the model then looks again at 12 frames sampled closely between the key frames before and after it. The end of the leg drive and the landing are placed by code, 0.1 s before and 0.6 s after contact: the model did not find them among close frames (the landing it picked with the player still in the air), and at 0.1 s before contact a dropped tossing arm shows best.
5. Two yes / no / unclear questions per category, each about one key frame: its phase's, or the one that shows it better (the tossing arm is judged at the trophy and at the end of the leg drive): one about the good position (`check` in `strokes/serve/criteria.yaml`) and one about the problem (`problem`). A check counts only when both agree; otherwise it is "unclear". The early drop of the tossing arm is asked differently: one neutral question (which way the tossing arm points, after finding the racket hand: straight up, a diagonal closer to vertical or to horizontal, horizontal, down) about the upper body only, cropped from a box the model draws and enlarged, because with yes / no the model agreed with whatever the question described. The earliest phase with an agreed problem is the point to improve, with its drill; with none, the answer is "no clear problem". One more call writes the observation, only about that point.

Every answer is JSON constrained by a schema, so the model can only choose known frames, phases, categories, and drills. Why each step exists is in the header of each prompt version, `strokes/serve/prompts/v*.yaml`.

In the app, upload one serve recorded from the side, following [docs/grabacion.md](docs/grabacion.md) (recording guide, in Spanish), or pick a demo video, and press "Analizar saque". Progress is shown in four numbered steps. The app shows:

- the phase with the main opportunity for improvement (and the phase whose frame shows it), the observation and a drill,
- how far to trust it ("Qué tan seguro está"): the confidence and how many checks the model could judge,
- the key frame of each of the six phases with its verdict in words and color: se ve bien (green), a mejorar (red), no se ve claro (grey), sin revisar; "★ Aquí" marks the frame the observation talks about. A frame opens large when tapped,
- the video in slow motion with the current phase,
- every check with what the model saw ("Qué revisamos en cada fase"),
- from the second attempt on, a comparison with the previous attempt.

![The app after analyzing a serve](docs/images/app/result.jpg)

"Terminar sesión" summarizes the session and proposes the focus of the next practice. With a player ID, the session is added to that player's history in `data/history/`. "Empezar de nuevo" clears the video, the answers and the session for the next player.

If the video cannot be analyzed (wrong angle, player out of frame, too dark, too short or too long, low frame rate, not a video), the app shows the reason and the recording tips instead of feedback. If the model gives no valid answer, the app says so and asks to try again. The page bundles its fonts and loads nothing from the internet, so it works offline.

### Demo mode (fair)

One analysis takes 15 to 45 s on the Mac. For the fair, put the demo videos in `data/demo/` (only players with `consent_image` = `yes`; include one bad recording to show the request for a new one) and run:

```bash
uv run python -m app.api --demo
```

At start, each video in `data/demo/` is analyzed once; the answer is saved in `data/processed/demo_cache/` and shown at once after that, marked "respuesta guardada". The videos appear under "Videos de la demo". Any other video, such as a visitor's, is analyzed live and its answer is not saved. A saved answer is reused only for the same video, model, prompt version and criteria version; delete `data/processed/demo_cache/` to analyze again.

## Evaluation

The labels written by the team are only the answer key: in Project 3 the model is not trained, it is used as downloaded with a prompt. See [what the labels are for](data/README.md#what-the-labels-are-for).

1. Put the videos in `data/raw/`. There is no coach: both members label each serve on their own, compare, and write the agreed labels in `eval/serve/labels.csv` (guide in [data/README.md](data/README.md)):

```bash
uv run python -m eval.label      # labeling tool, http://127.0.0.1:7861
uv run python -m eval.run_eval --agreement eval/serve/labels_ernesto.csv eval/serve/labels_hugo.csv
```

2. Run the full system and the two baselines:

```bash
uv run python -m eval.run_eval --variant full   --note "v4 prompt"
uv run python -m eval.run_eval --variant simple --note "baseline: frames only, no criteria"
uv run python -m eval.run_eval --model mlx-community/Qwen3-VL-2B-Instruct-4bit --note "baseline: smaller model"
```

Each run saves its predictions in `eval/serve/runs/` and appends a row to `experiments/iterations.csv`: phase accuracy, category F1, hallucination rate, correct re-recording requests, latency, tokens (every call), and peak memory. For the simple baseline the hallucination rate is an automatic proxy (the category belongs to another phase). For the full system the code chooses the category, so the rate comes from the members' review of each check next to its frame (tab "Revisar respuestas" of the labeling tool):

```bash
uv run python -m eval.run_eval --review eval/serve/runs/<run>.jsonl eval/serve/review_*.csv
```

Runs made before the labels existed (`--videos`) are scored afterwards against any label file, one system's runs together (JC and professionals):

```bash
uv run python -m eval.run_eval --score eval/serve/runs/<jc run>.jsonl eval/serve/runs/<pro run>.jsonl --labels eval/serve/labels.csv
```

The model samples its answers (temperature 0.2, `seed` in `configs/default.yaml`), so a run with another seed (`--seed`) can give other answers: the contact frame moves a few frames and an answer about the same frame can flip. One seed is not enough to compare two versions; compare the mean over several seeds:

```bash
uv run python -m experiments.seeds   # macro-F1 per prompt version and seed
```

With seeds 0 to 3, v21 has macro-F1 0.714 ± 0.233, v20 0.652 ± 0.178, v18 0.372 ± 0.144, v17 0.336 ± 0.168 and v12 0.210 ± 0.052. v21 finds JC's early tossing-arm drop on 66 of 80 serves (v20 58, v18 16) and gives 3 false alarms on 44 professional serves (v20 4, v18 2), all on one serve whose contact frame is placed late. Greedy decoding (`--temperature 0`) gives the same answers for every seed but scored lower (0.115 with v17).

Cost per analysis: the model runs on the Mac, so there is no API cost. Energy is not measured; as an estimate, about 30 s at an assumed 40 to 60 W is 0.3 to 0.5 Wh, a small fraction of a cent.

Settings that change results (number of frames, language model, prompt version, temperature) are in `configs/default.yaml`. Change one at a time and run the evaluation again.

## Privacy

- Videos and player histories stay in ignored folders under `data/`. Nothing is uploaded: the app listens only on `127.0.0.1`, and the Gradio fallback has its analytics disabled.

## Tests

```bash
uv run pytest
```

## Repository layout

```
configs/      shared settings (frames, model, generation parameters)
core/         code shared by all analyses: video, frames, llm, history
strokes/      one module per stroke; serve/ is the current prototype
app/          API and shared analysis (api.py, service.py), Gradio fallback (__main__.py)
web/          web page (React), served by app/api.py
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
