# Serve Lab

Interactive assistant that analyzes the **tennis serve** from a short video. It splits the motion into phases, finds the phase with the main opportunity for improvement, shows the frames that support the observation, and suggests a drill. It keeps a history of sessions to follow the player's progress.

Final Project of CC3092 Deep Learning y Sistemas Inteligentes (UVG), on the `proyecto-final` branch. The app runs the team's own model: MediaPipe finds the body in every frame and a small temporal CNN, trained on the team's labels, says whether the tossing arm drops early ([Final Project](#final-project-trained-models-and-their-value)). Project 3's system, where only a vision-language model looks at the video, is kept as version A to compare against; most of this README still documents it. Full proposal: [docs/proposal.md](docs/proposal.md).

## Summary of the work

The model (Qwen3-VL-4B, 4-bit, on the Mac) is never trained: the work is in the prompts and in the code around them, across prompt versions v4 to v22. They were measured on 31 serves filmed from the side: 20 of JC, an amateur whose tossing arm drops early, and 11 of three professionals with no clear problem. Claude labeled them as an LLM judge; the team's own labels are not compared yet. Each version fixed a failure seen in the runs:

- **v5 to v7:** the answer followed the order of the options. Then the model never said "no". Then it flagged every professional.
- **v8:** two opposite questions that must agree, and a crop around the player.
- **v9:** a close second look at the contact frame.
- **v13 and v17:** code places the leg drive 0.1 s before contact and the landing 0.6 s after it.
- **v18 and v22:** more tries when the phases look guessed.
- **v20 and v21:** the tossing arm. The model described JC's arm as "extended outward" but answered "above the head", because it looked at the racket hand. v20 makes it find the racket hand first and asks which way the other arm points. v21 counts a diagonal closer to horizontal as a drop.

The web page was also rebuilt in React with The Gridcn components.

### Results

Over sampling seeds 0 to 3, 4 runs per version (from `uv run python -m experiments.seeds`):

| Version | Change | Macro-F1 | JC drops found (of 80) | Professional false alarms (of 44) | Videos with no answer (of 128) |
| --- | --- | --- | --- | --- | --- |
| v12 | neutral question about the upper body | 0.210 ± 0.052 | 8 | 3 | 6 |
| v17 | landing placed 0.6 s after contact | 0.336 ± 0.168 | 15 | 2 | 6 |
| v18 | third try when the phases look guessed | 0.372 ± 0.144 | 16 | 2 | 2 |
| v19 | majority of three answers (not adopted) | 0.319 ± 0.084 | 10 | 2 | 2 |
| v20 | which way the tossing arm points | 0.652 ± 0.178 | 58 | 4 | 2 |
| v21 | a diagonal closer to horizontal is a drop | 0.714 ± 0.233 | 66 | 3 | 2 |
| **v22** | **fourth try (default)** | **0.842 ± 0.084** | **67** | **3** | **0** |

- **Baselines:** the simple prompt and the 2B model flag all 11 professionals and score 0.00 (seed 0).
- **What v22 still gets wrong:**
  - its 3 false alarms all come from one serve, pro03_s03, whose contact frame is placed late;
  - 13 of JC's 80 drops are still missed.
- **Speed:** one analysis takes about 44 s for a JC video and 29 s for a professional one on the Mac (mean over 4 seeds).

### Why Qwen3-VL-4B

The same prompt (v17) and seed (0) on three sizes of the same model family, all 4-bit on the Mac:

| Model | Macro-F1 | JC drops found (of 20) | Professional false alarms (of 11) | Seconds per JC video | Memory |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL-2B | 0.052 | 1 | 9 | 18 | 3.7 GB |
| **Qwen3-VL-4B** | **0.574** | **7** | **0** | 30 | 5.1 GB |
| Qwen3-VL-8B | 0.135 | 0 | 1 | 48 | 7.9 GB |

- The 2B model flags almost everyone, so it is the baseline that shows what the larger model adds.
- The 8B model is slower and needs more memory, and it was no better. With v6 and v7 it also followed the wording of the question: it flagged 9, then 11, of the 11 professionals (the 4B model with v6: 3).
- An open model that runs on the Mac keeps the players' videos on the computer, works without internet at the fair, and costs nothing per analysis. Its license is Apache-2.0, it reads several images in one request (32 frames to find the phases), and it fits in the Mac's 24 GB.
- Limits of this comparison: the prompts were written for the 4B model, so the 8B model might do better with prompts of its own. And it is one seed, while a version's score can change a lot between seeds.

Runs: 2B `eval/serve/runs/20261102-032818-full.jsonl` and `20261102-033105-full.jsonl`, 4B `20261102-023129-full.jsonl` and `20261102-023612-full.jsonl`, 8B `20261102-031416-full.jsonl` and `20261102-032147-full.jsonl`; with v6 and v7, `20261101-pro-v6-8b`, `20261101-pro-v7-8b`, `20261101-jc-crop-v6-8b` and `20261101-jc-crop-v7-8b`. Each is a row of `experiments/iterations.csv`.

### What the numbers mean

- **Precision**: when the app says "this is a problem", how often it is right. Low precision means false alarms: telling a player to fix something that was fine.
- **Recall**: of the real problems, how many the app finds. Low recall means missed problems: the player hears "no clear problem" while the arm drops.
- **F1**: one number that combines both, from 0 (useless) to 1 (always right). It is high only when both are high: flagging everyone finds every problem, and flagging no one gives no false alarm, but both get a low F1.
- **Macro-F1**: the F1 of each answer ("tossing arm drops early", "no clear problem") averaged with the same weight, so the rarer answer counts as much as the common one.
  - A system that always says "the arm drops" scores 0.39 here, and one that always says "no clear problem" scores 0.26. v22 scores 0.93 with seed 0.
  - A video with no answer counts as one more answer that is always wrong, so a single error costs about a third of macro-F1.
  - As a rough guide for this project: below 0.4 is no better than always giving the same answer, around 0.7 is useful but makes visible mistakes, and above 0.8 is good.
- **± (over 4 seeds)**: how much the score changes when the model samples differently. A large ± means a lucky or unlucky run could change the conclusion. v22's ± 0.084 is half of v20's 0.178.
- **Dev and test halves**: the videos are split in two, and a change that helps only on one half is suspect. Both halves were seen while choosing versions, so the scores are optimistic.
- **Professional false alarms and JC drops found**: the two kinds of mistake, counted directly. The first is precision on good serves, the second is recall on JC's problem. For a player, a wrong correction is worse than a missed one, so false alarms weigh more in the adoption rule.
- **Latency**: seconds per analysis, which is what a player waits.
- **Tokens**: how much the model reads and writes, which is the compute per analysis. It costs no money because it runs on the Mac.
- **Memory**: the RAM the model needs; it must fit on the Mac.
- **κ (kappa)**: agreement between two labelers beyond chance, from 0 (chance) to 1 (full agreement). Once the team writes its labels, κ will say how far Claude's labels can be trusted.

The model is never trained, so these numbers do not change it. They decide which prompt version the app uses. A version that finds more problems but gives more false alarms, like v20, is a trade-off the team has to choose.

### How decisions were made

- **Versions are files.** A version is never edited once it has results. Its header (`strokes/serve/prompts/v*.yaml`) says what changed, why, and what came out. Every run is a row of `experiments/iterations.csv`.
- **Versions are compared on 4 seeds.** The model samples its answers. Changing the seed moved the contact frame by up to 18 frames and changed answers about the same frame. So versions are compared on the mean over seeds 0 to 3, and on the dev and test halves separately.
- **A cheap test comes first.** An idea is tried first on saved frames (`experiments/probe_toss_arm.py cached`, about 3 minutes). Only then does it get a full run: about 80 minutes for 4 seeds.
- **The adoption rule.** A version becomes the default only when all of these hold:
  - its mean macro-F1 is higher;
  - its test half is no worse;
  - on professionals, it gives 0 false alarms with seed 0 and at most 2 of 44 in all;
  - it takes at most 45 s per JC video;
  - the tests pass;
  - the app still asks for a new recording on the bad demo video.
- **How the rule was applied:**
  - v19 varied less between seeds but scored lower, so v18 stayed.
  - v20 broke the false alarm limit (4 of 44), but it raised the mean macro-F1 from 0.372 to 0.652. The team chose to adopt it for its recall, and the exception is written in its header.
  - v21 and v22 each beat the version before on every measure.
- **v22 was measured without a full run.** Its extra try runs only after three guesses, so on every other video it makes exactly v21's calls. Only the two videos where v21 gave no answer were run again; `experiments/seeds.py` scores v21's runs with those two videos replaced.

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

The first analysis downloads MediaPipe's pose model (about 30 MB) into `models/`. Only version A, Project 3's system (`eval.run_eval`), downloads Qwen3-VL-4B (about 3 GB) into the Hugging Face cache. After that, everything runs offline on the Mac.

## Usage

```bash
uv run python -m app.api     # web app at http://127.0.0.1:7860
```

The page is a React app in `web/` (components from [The Gridcn](https://thegridcn.com), how to build and change it in [web/README.md](web/README.md)), served with its API by `app/api.py`.

How one video is analyzed (`app/service.py`, 3 to 6 s on the Mac):

1. OpenCV checks the video (frame rate, length).
2. MediaPipe Pose (heavy) finds 33 landmarks in every frame, inside a square around the region that moves (`core.pose`).
3. Contact is the highest wrist in the second after the toss; the 64 frames around it, from 1.6 s before to 0.5 s after, become the model's input (`strokes.serve.dataset`): the shoulders, elbows and wrists, with the tossing arm always on the same side.
4. The model of `configs/default.yaml` (M1c, one network per training seed) gives the probability that the tossing arm drops early; the drill is `toss_hold` from the threshold `eval/value.py` chose for the most value on validation scores, `steady_serves` below it (`strokes.serve.model`). Occlusion says which joint and moment the answer rests on: each arm joint, during each stretch of 0.27 s, is replaced by the same joint of a clean serve (the mean of the train serves without the problem) and the change in the probability is measured.

The video is recorded again when no person is found in most frames, when there is no toss (no wrist above the shoulders), or when MediaPipe saw less than 70 % of the arm points around contact; on the train split, 3 of the 7 serves the team could not judge fall under that line and no serve they judged. Under 85 %, the confidence is low.

The app shows:

- whether the tossing arm drops early, what the landmarks show (when the tossing wrist goes below its shoulder) and a drill,
- how far to trust it ("Qué tan seguro está"): the confidence (high when every seed agrees, MediaPipe saw the arms and the probability is 0.15 or more from the threshold; low when seeds disagree or under 85 % of the points were seen), the probability against the threshold and the spread over seeds, and the model's rates on the test videos,
- four key moments with the skeleton drawn and the tossing arm in orange: the start of the window, the tossing wrist at its highest, the last frame with it above the shoulder ("★ Aquí" when the arm drops early) and contact,
- "En qué se fijó el modelo": the occlusion map, joints by time, and the joint and stretch that move the answer most,
- the slow motion from 1.6 s before contact with the skeleton, the time from contact and the latest key moment,
- from the second attempt on, how the probability changed since the previous attempt.

"Terminar sesión" summarizes the session from the attempts alone (how often the arm dropped early, how the probability moved) and gives the drill for the next practice. With a player ID, the session is added to that player's history in `data/history/`. "Empezar de nuevo" clears the video, the answers and the session for the next player.

### Version A: the Project 3 pipeline (Qwen3-VL)

How one video was analyzed in Project 3 (prompt v22, `uv run python -m eval.run_eval`):

1. OpenCV checks the video (frame rate, length) and takes 32 evenly spaced frames.
2. The model draws a box around the player on the first, middle and last frames. The box is joined, grown to the reach of a serve (toss, racket, landing), and every frame is decoded again cropped to it, so a player far from the camera fills the frame. No crop when it would keep most of the frame.
3. The model decides whether the video can be analyzed and where the serve starts and ends. Players bounce the ball before serving, so most of a video is not the serve.
4. OpenCV takes 32 frames again, only inside the serve, and the model picks the frame that best shows each of the six phases. Each frame has its number drawn on it. When the answer is six frames in a row (the model counted instead of looking), it tries again on every 2nd frame, then on every 3rd, then on every 4th. Contact lasts a few hundredths of a second, so the model then looks again at 12 frames sampled closely between the key frames before and after it. The end of the leg drive and the landing are placed by code, 0.1 s before and 0.6 s after contact: the model did not find them among close frames (the landing it picked with the player still in the air), and at 0.1 s before contact a dropped tossing arm shows best.
5. Two yes / no / unclear questions per category, each about one key frame: its phase's, or the one that shows it better (the tossing arm is judged at the trophy and at the end of the leg drive): one about the good position (`check` in `strokes/serve/criteria.yaml`) and one about the problem (`problem`). A check counts only when both agree; otherwise it is "unclear". The early drop of the tossing arm is asked differently: one neutral question (which way the tossing arm points, after finding the racket hand: straight up, a diagonal closer to vertical or to horizontal, horizontal, down) about the upper body only, cropped from a box the model draws and enlarged, because with yes / no the model agreed with whatever the question described. The earliest phase with an agreed problem is the point to improve, with its drill; with none, the answer is "no clear problem". One more call writes the observation, only about that point.

Every answer is JSON constrained by a schema, so the model can only choose known frames, phases, categories, and drills. Why each step exists is in the header of each prompt version, `strokes/serve/prompts/v*.yaml`.

In the Project 3 app, the answer was the phase to improve among six, with each phase's key frame and verdict ([screenshot](docs/images/app/result.jpg)).

In the app, upload one serve recorded from the side, following [docs/grabacion.md](docs/grabacion.md) (recording guide, in Spanish), or pick a demo video, and press "Analizar saque". Progress is shown in four numbered steps. If the video cannot be analyzed, the app shows the reason and the recording tips instead of feedback. The page bundles its fonts and loads nothing from the internet, so it works offline.

### Demo mode (presentation)

For a presentation, put the demo videos in `data/demo/` (only players with `consent_image` = `yes`; include one bad recording to show the request for a new one) and run:

```bash
uv run python -m app.api --demo
```

At start, each video in `data/demo/` is analyzed once; the answer is saved in `data/processed/demo_cache/` and shown at once after that, marked "respuesta guardada". The videos appear under "Videos de la demo". The poster for the stand is in `docs/poster/`: `poster.html` to show on a screen (it needs no internet) and `poster.png`. The PDF to print at 24 × 36 in is built locally and not kept in Git. Any other video, such as a visitor's, is analyzed live and its answer is not saved. A saved answer is reused only for the same video, model weights, threshold and pose model; delete `data/processed/demo_cache/` to analyze again.

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

With seeds 0 to 3, v22 has macro-F1 0.842 ± 0.084 and no video without an answer (128 analyses), v21 0.714 ± 0.233, v20 0.652 ± 0.178, v18 0.372 ± 0.144, v17 0.336 ± 0.168 and v12 0.210 ± 0.052. v21 finds JC's early tossing-arm drop on 66 of 80 serves (v20 58, v18 16) and gives 3 false alarms on 44 professional serves (v20 4, v18 2), all on one serve whose contact frame is placed late. Greedy decoding (`--temperature 0`) gives the same answers for every seed but scored lower (0.115 with v17).

Cost per analysis: the model runs on the Mac, so there is no API cost. Energy is not measured; as an estimate, about 30 s at an assumed 40 to 60 W is 0.3 to 0.5 Wh, a small fraction of a cent.

Settings that change results (number of frames, language model, prompt version, temperature) are in `configs/default.yaml`. Change one at a time and run the evaluation again.

### Final Project: trained models and their value

The Final Project puts models the team trains at the core, on MediaPipe landmarks (`uv run python -m core.pose`), and keeps the Project 3 system as the reference (version A). The label is one fault: whether the tossing arm drops early (`data/README.md`).

```bash
uv run python -m strokes.serve.dataset                        # 64-frame windows around contact, eval/serve/labels.csv
uv run python -m strokes.serve.baseline                       # B0, a rule on the tossing wrist
uv run python -m strokes.serve.train                          # M1, temporal CNN on the 17 joints
uv run python -m strokes.serve.train --arms                   # M1b, the arms only
uv run python -m strokes.serve.train --arms --epochs 85 --name m1c   # M1c, M1b stopped where validation loss is lowest
uv run python -m eval.value                                   # value of each model in quetzales (configs/value.yaml)
uv run python -m strokes.serve.model                          # occlusion map of the app's model (M1c) on the test videos
```

Validation scores come from 5-fold cross-validation on the train split; the test split (JC's serves of Nov 1 and professional pro01) is never used to choose. Scores, loss curves, the value curve and the sensitivity table are written to `experiments/serve/`. Results on the 18 test videos (5 seeds for the CNNs; value per analysis with the threshold chosen on validation, 95 % bootstrap interval):

| Model | Test macro-F1 | Test AUROC | Q per analysis | Seconds |
|---|---|---|---|---|
| A: Qwen3-VL-4B, Project 3 (v22), one seed | 0.625 | 0.646 | 9.09 (−5.44 to 23.62) | 49.0 |
| B0: rule | 0.723 | 0.800 | 16.25 (4.85 to 27.42) | 5.0 |
| M1: CNN, 17 joints, 300 epochs | 0.482 ± 0.090 | 0.532 ± 0.057 | 12.00 (12.00 to 12.00) | 5.0 |
| M1b: CNN, arms only | 0.682 ± 0.039 | 0.880 ± 0.034 | 16.65 (9.99 to 24.16) | 5.0 |
| M1c: M1b, 85 epochs | 0.611 ± 0.055 | 0.757 ± 0.023 | 18.84 (12.00 to 26.44) | 5.0 |

Recommending the drill to everyone, with no model, is worth Q12.00 per analysis; M1's value-optimal threshold does exactly that. The intervals overlap: 18 test videos with 5 negatives cannot separate the models yet, which is why more amateur players are being recorded. The coach price and the hit, miss and false-alarm weights are estimates until the academy quotes and the improvement study replace them.

The app runs M1c, the model with the most value on validation scores, at the threshold chosen there (0.198). Its occlusion map on the test videos (`experiments/serve/m1c-occlusion.png`) shows where the answer rests: on serves where the arm drops early, mostly on the tossing wrist from 0.53 to 0.27 s before contact and at the start of the window, 1.6 s before contact, then on the tossing elbow; the racket arm barely moves it. Hiding the points instead, as MediaPipe does when it loses one, mostly raises the probability: the model reads an unseen arm point as a sign of the fault, a bias to watch with new players and cameras.

## Privacy

- Videos and player histories stay in ignored folders under `data/`. Nothing is uploaded: the app listens only on `127.0.0.1`, and the Gradio labeling tool (`eval/label.py`) has its analytics disabled.

## Tests

```bash
uv run pytest
```

## Repository layout

```
configs/      shared settings (frames, model, generation parameters)
core/         code shared by all analyses: video, frames, llm, history
strokes/      one module per stroke; serve/ is the current prototype
app/          API and the analysis of one video with the team's model (api.py, service.py)
web/          web page (React), served by app/api.py
data/         videos (not in Git) and annotations, see data/README.md
eval/         evaluation sets and metric scripts, per stroke
experiments/  iteration log (iterations.csv)
docs/         proposal, licenses, images, fair poster
tests/        automated checks
```

Rules that keep the repository scalable:

- `core/` contains no stroke-specific logic. `tests/test_architecture.py` fails if `core/` imports `strokes`.
- Each stroke lives in its own folder under `strokes/` with the same structure.
- Criteria, drills, and prompts are versioned files, not hard-coded strings.
- Each stroke has its own evaluation set under `eval/`.

## Licenses

Third-party models, libraries, and datasets: [docs/licenses.md](docs/licenses.md). The project code has no license yet (all rights reserved).
