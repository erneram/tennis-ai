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
- [ffmpeg](https://ffmpeg.org/), only to cut long practice sessions into serves (`brew install ffmpeg`).

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
3. Contact is the highest wrist in the second after the toss; the 64 frames around it, from 1.6 s before to 0.5 s after, become the model's input (`strokes.serve.dataset`): the height of each elbow and wrist above its own shoulder, tossing arm first (`strokes.serve.train.heights`).
4. The model of `configs/default.yaml` (M1e, one network per training seed) gives the probability that the tossing arm drops early; the drill is `toss_hold` from the threshold `eval/value.py` chose for the most value on validation scores, `steady_serves` below it (`strokes.serve.model`). Occlusion says which joint and moment the answer rests on: each elbow and wrist height, during each stretch of 0.27 s, is replaced by the same one of a clean serve (the mean of the train serves without the problem) and the change in the probability is measured.

The video is recorded again when no person is found in most frames, when there is no toss (no wrist above the shoulders), when the camera is not to the side (the shoulders less than 0.4 torso lengths apart across the image in the stance, see bias and robustness below), or when MediaPipe saw less than 70 % of the arm points around contact; on the train split, 3 of the 7 serves the team could not judge fall under that line and no serve they judged. Under 85 %, the confidence is low.

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

1. Put the videos in `data/raw/`, one serve each. A long practice session with many serves goes to `data/raw/uncut/` and is cut first (needs [ffmpeg](https://ffmpeg.org/)): `find` writes the candidates and an image of each in `data/processed/split/`; look at the image, then `cut` with the numbers that are not serves (a bag MediaPipe took for a person, a toss caught without a swing) and the seconds of any serve it missed. Clips are 5 s, from 3 s before the top of the toss, with the phone's rotation applied; `cuts.csv` next to the session records where each comes from. On the first 10 sessions (39 minutes) `find` gave 169 candidates for the 102 clips kept by checking each one and every stretch where it lost the player, and missed one serve; when labeling, the team found 4 of the 102 clips without a serve (a toss with no swing). MediaPipe follows one person, and with someone else on court it can pick the wrong one (in ea-3-s05 and s06 the person nearer the camera): crop such a clip to the player's side (`ffmpeg ... -vf crop=480:1920:0:0`) and check the skeleton with `uv run python -m core.pose <clip> --sheet` before labeling.

```bash
uv run python -m strokes.serve.split find data/raw/uncut/*.MOV
uv run python -m strokes.serve.split cut data/raw/uncut/ea_2.MOV --name ea-2 --drop 1,5,8 [--add 262.0]
```

A video already slowed down (slow motion played back at 24 or 30 fps) is cut with `retime`: give how many times slower than real it plays and the seconds of each serve in it, and each serve is written at real speed and at most 60 fps, so the windows have the same timing as the other clips; `retimed.csv` next to the video records them. When the video does not say how much it was slowed down, the tossed ball tells: in free fall it accelerates at g, so in the video at g/k² for a factor k, with the torso (about 0.53 m) as the ruler. On the Federer videos this gave 9.3 to 9.7 (a 240 fps camera played at 24 fps: 10) and 7.8 (240 at 30 fps: 8).

```bash
uv run python -m strokes.serve.split retime data/raw/roger/side.mp4 --name pro04 --slow 8 --spans 0-32.5
```

There is no coach: both members label each serve on their own, compare, and write the agreed labels in `eval/serve/labels.csv` (guide in [data/README.md](data/README.md)). The labeling tool lists only the videos the member has not labeled yet, new clips included; clips already in `labels.csv` that the team asked someone else to label (Federer's, pro04) are not listed, and saving the agreement keeps them:

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
uv run python -m strokes.serve.pretrain --epochs 2000         # M2's encoder, pretrained without labels (THETIS, below)
uv run python -m strokes.serve.train --arms --init pretrain   # M2, M1b fine-tuned from that encoder
uv run python -m strokes.serve.train --arms --init pretrain --epochs 58 --name m2c   # M2c, M2 stopped where validation loss is lowest
uv run python -m strokes.serve.train --heights --epochs 85 --name m1e   # M1e, M1c on the arm heights only (the app's model)
uv run python -m strokes.serve.train --heights --epochs 85 --more jc-post --name m1f   # M1f, M1e with JC's serves after the feedback
uv run python -m strokes.serve.train --heights --epochs 85 --synthetic --name m1g   # M1g, M1e with synthetic faults
uv run python -m eval.value                                   # value of each model in quetzales (configs/value.yaml)
uv run python -m strokes.serve.model --name m1e               # occlusion map of the app's model on the test videos
uv run python -m eval.eda                                     # the data: labels, recordings, landmarks
uv run python -m eval.bias                                    # errors by group, unseen points, mirror, camera in front, new players (after eval.eda)
uv run python -m eval.improvement                             # how far JC moved toward the professionals after the feedback
```

M2's pretraining reads THETIS ([github.com/THETIS-dataset](https://github.com/THETIS-dataset), Gourgari et al., CVPR Workshops 2013): 495 flat, kick and slice serves of 55 players (31 beginners), filmed indoors from the front without a ball. They are downloaded into the ignored `data/external/thetis/` and prepared once (about 20 minutes; licensing in [docs/licenses.md](docs/licenses.md)):

```bash
mkdir -p data/external/thetis && for d in flat_service kick_service slice_service; do
  gh api "repos/THETIS-dataset/dataset/contents/VIDEO_RGB/$d" --jq '.[].download_url'
done | (cd data/external/thetis && xargs -n 1 -P 8 curl -sSfLO)
uv run python -m core.pose data/external/thetis/*.avi --out data/processed/thetis-pose
uv run python -m strokes.serve.dataset --pose data/processed/thetis-pose --out data/processed/thetis_dataset.npz
```

The data (`eval/eda.py`, `experiments/serve/eda.csv` and `eda.png`): 55 videos, 44 of JC, the only amateur, and 11 of three professionals. The team judged 48: JC's 37 split into 23 where the tossing arm drops early and 14 with no clear problem, and all 11 professional serves have no clear problem, so every serve with the fault is JC's. The two raters agreed on which videos can be judged (kappa 1.00) and less on the category (kappa 0.50 on the 48 both judged; the 12 disagreements were settled together by watching the videos). The recordings follow the player: JC's judged serves are landscape at 60 fps, 9 of the 11 professional ones are at 30 fps and 10 are portrait. MediaPipe saw 90 % of JC's arm points and 95 % of the professionals'; 0.77 to 0.98 on judged serves and 0.58 to 0.82 on the 7 the team could not judge, where the app's 70 % line comes from. JC tosses with the right arm, the professionals 8 times with the right and 3 with the left. B0's measure, when the tossing wrist goes below its shoulder, has the same median for JC with and without the fault (0.17 s before contact) and 0.10 s for the professionals, whose tossing wrist also rises later and higher: the measure separates JC from the professionals more than it separates the fault within JC.

Validation scores come from 5-fold cross-validation on the train split (30 videos); the test split (JC's serves of Nov 1 and professional pro01, 18 videos) is never used to choose. Scores, loss curves, the value curve and the sensitivity table are written to `experiments/serve/`. Results (5 seeds for the CNNs; value per analysis with the threshold chosen on validation, 95 % bootstrap interval on test):

| Model | Validation AUROC | Validation Q | Test macro-F1 | Test AUROC | Test Q per analysis | Seconds |
|---|---|---|---|---|---|---|
| A: Qwen3-VL-4B, Project 3 (v22), one seed | 0.650 | 26.16 | 0.625 | 0.646 | 15.91 (−9.52 to 41.34) | 49.0 |
| B0: rule | 0.608 | 24.50 | 0.723 | 0.800 | 28.43 (8.48 to 47.99) | 5.0 |
| M1: CNN, 17 joints, 300 epochs | 0.706 ± 0.044 | 22.99 | 0.482 ± 0.090 | 0.532 ± 0.057 | 21.00 (21.00 to 21.00) | 5.0 |
| M1b: CNN, arms only, 300 epochs | 0.721 ± 0.042 | 27.49 | 0.682 ± 0.039 | 0.880 ± 0.034 | 29.14 (16.45 to 42.44) | 5.0 |
| M1c: M1b, 85 epochs | 0.681 ± 0.063 | 32.04 | 0.611 ± 0.055 | 0.757 ± 0.023 | 32.97 (21.00 to 46.27) | 5.0 |
| M2: M1b from the pretrained encoder, 300 epochs | 0.673 ± 0.036 | 29.12 | 0.700 ± 0.061 | 0.702 ± 0.033 | 34.30 (21.00 to 47.60) | 5.0 |
| M2c: M2, 58 epochs | 0.644 ± 0.031 | 35.63 | 0.718 ± 0.037 | 0.618 ± 0.038 | 32.97 (21.00 to 46.27) | 5.0 |
| M1d: M1c plus JC's 44 serves after the feedback | 0.858 ± 0.022 | 43.17 | 0.546 ± 0.060 | 0.689 ± 0.020 | 10.51 (−12.80 to 32.58) | 5.0 |
| M1e: M1c on the arm heights only (the app's model) | 0.790 ± 0.065 | 36.00 | 0.695 ± 0.157 | 0.895 ± 0.043 | 30.62 (20.37 to 41.11) | 5.0 |
| M1f: M1e plus JC's 44 serves after the feedback | 0.869 ± 0.038 | 31.98 | 0.664 ± 0.171 | 0.905 ± 0.065 | 25.77 (15.36 to 36.10) | 5.0 |
| M1g: M1e plus synthetic faults | 0.742 ± 0.061 | 36.96 | 0.769 ± 0.069 | 0.822 ± 0.102 | 30.47 (18.96 to 42.28) | 5.0 |

Recommending the drill to everyone, with no model, is worth Q21.00 per analysis; M1's value-optimal threshold does exactly that. The intervals overlap: 18 test videos with 5 negatives cannot separate the models yet, which is why more amateur players are being recorded. The coach price is the middle of the academy quotes, Q200 to 500 an hour (the value scales with it, the thresholds and the ranking do not, and the sensitivity at ±50 % covers the range); the hit, miss and false-alarm weights are estimates until the improvement study replaces them. The share of serves with the fault matters most: 62 % is JC's before the feedback. M1c is worth more than B0 only when more than about half the serves have the fault (51 % with the test rates, 74 % with the false alarms on EA, see bias below); with the share among the amateurs we recorded before any feedback, 23 of 91 serves (25 %), B0, M1c, M2c and Qwen are worth no more on validation than never recommending the drill (M1, M1b and M2 are, at thresholds above 0.6).

Pretraining against training from scratch (M2 against M1b, M2c against M1c: same data, seeds, folds and training) does not improve the ranking (validation AUROC 0.673 against 0.721) but gives the most value on validation (M2c, Q35.63 against M1c's Q32.04). It is a weak pretraining: filling in hidden stretches of the arms, the encoder errs 0.026 torso lengths² on held-out THETIS players after 2000 epochs, twice as much as a straight line across each stretch (0.013, `experiments/serve/pretrain-curves.png`), so it learns little beyond smooth motion; THETIS is also filmed from the front, not from the side. A first try with 300 epochs of pretraining (error 0.038) gave M2 a validation AUROC of 0.630, and 2000 epochs were kept on validation scores.

Until Nov 8 the app ran M1c at the threshold chosen on validation scores (0.198). The rule fixed beforehand, the most value on validation, picked M2c (Q35.63 against Q32.04), and the app ran it until the bias check below: on JC's serves alone M2c ranks the fault worst of the models (AUROC 0.36 on validation and on test) and calls "drops early" on every JC serve without the fault, so its extra value comes from recommending the drill to JC, not from seeing the fault. M1c is less tied to the player (0.44 and 0.60), with the same test value (Q18.84) and the same test rates. Its occlusion map on the test videos (`experiments/serve/m1c-occlusion.png`) shows where the answer rests: on serves where the arm drops early, on the tossing wrist from 0.53 to 0.27 s before contact and at the start of the window, 1.6 s before contact; the racket arm barely moves it. M2c's map (`m2c-occlusion.png`) rests on the same wrist through the whole drop. Hiding the points instead, as MediaPipe does when it loses one, mostly raises M1c's probability: the model reads an unseen arm point as a sign of the fault (measured below).

**M1e: the arm heights only (the app's model since Nov 8).** The labeling question asks whether the tossing hand is still up when the racket is lowest. Measured directly, with no training (the height of the tossing wrist above its shoulder at the frame where the racket wrist is lowest, in the 0.6 s before contact), it orders the players as a coach would: medians of 0.40 torso lengths for JC with the fault, 0.57 for JC without it, 0.66 for JC after the feedback, 0.91 for EA, 1.07 for the professionals and 1.09 for Federer. Within JC it ranks the fault with AUROC 0.82 on the train split and 1.00 on the 16 test serves (B0's measure 0.41 and 0.83). M1c sees the same arms, but also where they are across the image and which points MediaPipe saw, and those tell players and recordings apart. M1e is M1c (same network, seeds, folds and 85 epochs) whose input is only the height of each elbow and wrist above its own shoulder; one configuration was trained and nothing was tuned. Against M1c, validation AUROC is 0.79 against 0.68, and on JC's serves alone 0.62 against 0.44 on validation and 0.84 against 0.60 on test. It has the most value on validation of all the models (Q36.00; M2c Q35.63, M1c Q32.04), so the rule fixed beforehand picks it; on test it is worth Q30.62 against Q32.97, inside each other's intervals. Its occlusion map (`m1e-occlusion.png`) rests on the racket wrist 1.07 to 0.80 s before contact, when the racket arm starts to go up, and on the tossing wrist 0.53 to 0.27 s before contact. In a one-off ablation with the same training (not in the repository; test AUROC of the mean score), the heights gave 0.91 without the mask and 0.88 with it, and the arms' height in the image (y without x, not measured against the shoulder) 0.74, like M1c's 0.75: the gain comes from measuring each joint against its own shoulder. Caveat: M1e was designed on Nov 8, after the new players had shown that the earlier models do not carry over, and the measure above was computed once on every split, test included. Nothing was tuned on test or on the new players, but they are no longer data nobody had looked at.

M1f is M1e trained with JC's 44 serves after the feedback too, as M1d is M1c with them. Unlike M1d it keeps the ranking (test AUROC 0.91, 0.87 on JC's serves alone) and calls "drops early" on 4 % of EA's serves, but at its threshold it finds fewer of the test faults (recall 85 %, Q25.77): with the heights only, those serves no longer teach the session, but 74 videos with 10 faults are still too few to choose it over M1e.

M1g is M1e trained with synthetic faults too (`--synthetic`). In each fold, each training serve without the fault gets a copy where the tossing arm drops early: its elbow and wrist are warped in time so the wrist goes below its shoulder 0 to 5 frames before the racket wrist is lowest, and the racket arm is left as it was (`train.synthetic`). 11 of the 20 clean training serves give one (the others drop the arm only after contact), from JC, pro02 and pro03; validation and test are scored on real serves only. The aim was serves with the fault that are not only JC's. The rule fixed before training (more value on validation than M1e, and a within-JC validation AUROC no lower than M1e's 0.62) rejects it. Its validation value is higher (Q36.96 against Q36.00), but within JC it ranks no better than chance (0.53). On serves it never trained on it also calls "drops early" more often: JC after the feedback 84 % against 66 %, EA 4 % against 2 %, THETIS 65 to 69 % against 40 to 46 %. The copies teach the drop as the code draws it and push the model toward recommending the drill; they do not stand in for real faults of other players.

Bias and robustness (`eval/bias.py`, `experiments/serve/bias.csv` and `bias.png`), for B0, M1c, M2c and later M1d, M1e and M1f at their thresholds:

- **Amateur against professional, not the fault.** On JC's serves alone the models rank no better than chance on validation (AUROC B0 0.41, M1c 0.44, M2c 0.36; 10 serves with the fault, 11 without); on test, with only 3 serves without it, B0 0.83, M1c 0.60, M2c 0.36. On validation and test together they call "drops early" on most of JC's serves without the fault (B0 79 %, M1c 97 %, M2c 100 % of 14) and on few professional ones (B0 36 %, M1c 9 %, M2c 4 % of 11). With one amateur who has every serve with the fault, the models mostly learned to tell JC from the professionals, and their value comes from recommending the drill to JC. Frame rate and orientation follow the player, so they cannot be separated from it. The fix is data: more amateurs, with and without the fault, recorded the same way (the improvement study's videos).
- **Fewer points seen.** With 10 to 40 % of the arm points of the test windows set unseen and interpolated, as with worse light or a farther camera, M1c's mean probability rises from 0.49 to 0.62 and it calls "drops early" on 94 % of the serves instead of 89 %; M2c stays between 0.56 and 0.59 (89 %) and B0 hardly moves. Pretraining, which hides stretches of the arms, made M2c robust to missing points. The app also asks for a new recording under 70 % of the points seen and lowers the confidence under 85 %.
- **Left-handed players.** The arm models put the tossing arm on the same side, so a mirrored window gives the same probability (largest change 0).
- **Camera in front of the player.** On THETIS, filmed from in front of the player, the models call "drops early" on 73 to 93 % of the serves, experts included (B0 82 % of beginners and 93 % of experts, M1c 77 and 73 %, M2c 89 and 87 %): they never saw that view. The app now checks the camera before it answers: when the shoulders are less than 0.4 torso lengths apart across the image in the stance (the first 0.53 s of the window, `strokes.serve.model.shoulder_width`), one shoulder hides the other and the video is recorded again. On our videos only 5 the team could not judge fall under the line (0.17 to 0.36; every judged serve 0.47 or more); on THETIS, 79 % of the serves. The 21 % left still get the same false alarms (M1c 69 %, M2c 87 %), so the recording guide stays the main defense.
- **Players recorded after the models were chosen.** On Nov 7, with the models and thresholds already fixed, two sessions were cut into clips (split `new` of `strokes/serve/dataset.py`, never used to train, pretrain or choose): a new amateur, EA, and JC after the feedback. The two raters agreed on all of them except 4 clips of JC with no serve, settled together, and judged all 98 serves without the fault: EA 54, JC 44. The file names said which session each clip came from, so these labels are not blind. Scored as they are, the models call "drops early" on EA's serves B0 13 %, M1c 80 %, M2c 57 %, and on JC's after the feedback B0 70 %, M1c and M2c 100 %; the app sends back only 3 of EA's. The trained models do not carry over to a new player and do not see JC's change; B0, a single hand-made measure, carries over best to EA. A fourth professional, Federer (pro04, 8 serves from slowed-down videos found online, played at real speed with `retime`), was labeled by Claude at the team's request, not watched by the team: on all 8 the tossing hand is still above the head when the racket is lowest. M1c and M2c call none of them "drops early", M1d 1, B0 5 (62.5 %): his tossing wrist goes below the shoulder 0.10 to 0.13 s before contact (3 and 4 frames at 30 fps; the other professionals' median is 0.10 s), right on the rule's line at 0.133 s, so B0 calls a professional serve early by a single frame. That measure hardly moves for JC (median 0.15 s before contact after the feedback, 0.17 s before it), while the mean height of his tossing wrist after the feedback stays up longer, like his serves without the fault (`eda.png`).
- **More data without the fault (M1d).** M1d is M1c trained with JC's 44 serves after the feedback too (`--arms --epochs 85 --more jc-post`); EA stays unseen. Its validation scores (AUROC 0.86, Q43.17) include those 44 serves, which it tells apart easily, so they do not compare with the other models'. It calls "drops early" on 17 % of EA's serves (M1c 80 %) and on 1 % of the THETIS beginners and 13 % of the experts, but on test it finds 71 % of the faults (M1c 100 %) and is worth Q10.51 (M1c Q32.97), and on JC's serves before the feedback it calls "drops early" on 91 % of the 14 without the fault and 83 % of the 23 with it. Every serve with the fault was filmed from one side of the player (the tossing arm on the right of the image) and every serve of Nov 7 from the other (on the left), and the arm input mirrors the windows so the tossing arm is always on the same side, which evens out the side itself; where the arms are across the image and which points MediaPipe saw still differ by session. M1d most likely learned the session, not the fault: M1f, the same data with the heights only, keeps the ranking. It does not replace M1c. The data that would help are serves with the fault filmed the way the Nov 7 sessions were.
- **Heights only (M1e, M1f).** On players never trained on, M1e calls "drops early" on 2 % of EA's serves (M1c 80 %, B0 13 %), none of Federer's, 66 % of JC's after the feedback (M1c 100 %, B0 70 %) and 46 % of the THETIS beginners and 40 % of the experts (M1c 77 and 73 %); M1f on 4 % of EA's. With 10 to 40 % of the arm points unseen, M1e's mean probability stays at 0.57 (it gets no mask). On JC's serves without the fault it still recommends the drill on 89 % (validation and test, 14 serves), so within JC it ranks better than M1c but its threshold still sits below most of his serves.
- **How far JC moved toward the professionals (`eval/improvement.py`, `experiments/serve/improvement.csv`).** The labeling question as a number: the height of the tossing wrist above its shoulder when the racket wrist is lowest (relative to its own shoulder) in the 0.6 s before contact, read from M1e's input with no training and no labels. Within JC's serves before the feedback it ranks the fault with AUROC 0.85 on train and 1.00 on test (median 0.31 torso lengths with the fault, 0.57 without). Its median is 0.38 before the feedback (37 serves), 0.69 after it (44; 0.56 to 0.75 across the four videos of Nov 7), 0.91 for EA and 1.05 for the professionals (29 serves of pro01 to pro05, Federer's included): JC closed 47 % of the gap to the professionals (95 % interval 31 to 56 %). The app's answer moves about as much: M1e sends 97 % of his serves before the feedback to the drill, 66 % after it and none of the professionals' (32 % of the gap, 18 to 47 %). 34 % of his serves after the feedback reach the professionals' range (at or above their 10th percentile, 0.73 with pro05's ten professionals; 0.83 and 11 % with the 19 serves before them), against none before and 96 % of EA's. The team's labels say all of the way (62 % with the fault before, none after), but the labels after the feedback are not blind. Caveats: one player and one session after the feedback, so the interval, which resamples serves, is too narrow; that session also changed the camera side and orientation, which do not move the measure among the professionals (median 1.03 with the tossing arm on the left of the image, 1.06 on the right) but cannot be ruled out for JC; and the number depends on how the measure is read: with the racket wrist lowest in the image instead of relative to its shoulder, 31 % (19 to 41 %). So one session of feedback took JC about a third to a half of the way to a professional's toss, not yet to a professional's serve. The improvement study's amateurs, measured before and after, would tell whether that holds for other players.
- **First fresh test: the YouTube clips of Nov 9.** M1e was chosen on Nov 8; these clips were cut on Nov 9 (`docs/licenses.md`) and labeled after. Ernesto labeled them alone at the team's request (`ONE_RATER` in `eval/label.py`): 13 of the 104 clips can be judged, the rest are filmed from behind or from a back diagonal. Claude's labels, a third rater, agree on which clips can be judged with kappa 0.31, almost all of it on one lesson (yt2-c) Ernesto saw as filmed from behind. M1e calls "drops early" on none of the 10 serves of pro05, each a different professional, and on neither of the 2 amateur serves of yt2-a without the fault (B0 on all 12, M1c on 2 and 1, M2c on 3 and 2). It calls the one serve with the fault, yt3-s07, "drops early", but the app sends it back: the camera is in front of the player (shoulders 0.06 torso lengths apart, the line is 0.4). So M1e carries over to ten new professionals, while whether it finds the fault in an amateur other than JC is still untested.

**Limitation: what the app's answer means.** M1e is the first model that carries over to players it was not trained on (2 % false alarms on EA, none on Federer), but every serve with the fault still comes from one amateur, only 3 of JC's test serves are without it, and on JC's serves without the fault it recommends the drill 89 % of the time; on JC's after the feedback, 66 %. So "drops early" means "the tossing hand comes down sooner than in the serves the model learned as clean", worth trying the drill, and the confidence matters: next to every answer the app shows the model's test rates (the drill recommended on 56 % of the test serves without the fault, 5 serves) and the confidence. M1e was chosen after the new players were seen (above); its first fresh test, the YouTube clips of Nov 9, has no false alarms on ten new professionals, but its one amateur serve with the fault was filmed from the front and sent back. What would lift the limitation: the improvement study's amateurs, with and without the fault, recorded the same way; then retraining, and the within-player check of `eval/bias.py` again.

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
