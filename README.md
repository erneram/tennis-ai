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
4. The model of `configs/default.yaml` (M1e, one network per training seed) gives the probability that the tossing arm drops early, a point to improve from the threshold `eval/value.py` chose for the most value on validation scores (`strokes.serve.model`). Occlusion says which joint and moment the answer rests on: each elbow and wrist height, during each stretch of 0.27 s, is replaced by the same one of a clean serve (the mean of the train serves without the problem) and the change in the probability is measured.
5. The rest of the body: the pose measures of `strokes.serve.body` and the body models named in `experiments/serve/points.csv` (M3b for the landing, M4 for the legs) answer for the legs, the racket drop, the arm at contact and the landing. `app.service.points` turns all of it into up to three points to improve, each with its drill; with none, the drill is `steady_serves` (see "More than one point to improve" below).

The video is recorded again when no person is found in most frames, when there is no toss (no wrist above the shoulders), when the camera is not to the side (the shoulders less than 0.4 torso lengths apart across the image in the stance, see bias and robustness below), or when MediaPipe saw less than 70 % of the arm points around contact; on the train split, 3 of the 7 serves the team could not judge fall under that line and no serve they judged. Under 85 %, the confidence is low.

The app shows:

- a title that names the points to improve ("A mejorar: brazo al impacto, aterrizaje y brazo que lanza") or says "No vemos un problema claro", and how many parts of the serve were looked at,
- up to three points to improve, the one that was right most often on validation first, in plain words and without measures: what the app saw ("Le pegas a la pelota más abajo de lo que alcanzas…"), what to aim for ("Lo ideal: …"), a badge that says in words how far to trust the point, from how often it was right on validation ("Muy probable" from 90 %, "Probable" from 70 %, "En duda" under that), its drill and what the badge means ("Cuando la app marca esto, suele tener razón, pero no siempre.") with one sentence for its level of evidence; the measures in torso lengths, the professionals' range, the level of evidence and the validation rates are in a folded "Detalle técnico" on each point,
- "Lo demás que revisamos": the other parts looked at, each one fine, inside or outside the professionals' range (a comparison with no drill) or not seen in the video, and the aspects the app does not evaluate (the shoulders, the racket when the ball is at its peak),
- up to five key moments, cropped to the player, with the skeleton drawn and the part each one is about in orange: the knees most bent, the tossing wrist at its highest, the last frame with it above the shoulder (only when the arm drops early), contact and, when the video goes on, the landing; "★ Aquí" marks the ones that show a point given,
- the slow motion from 1.6 s before contact with the skeleton, the time from contact and the latest key moment,
- "Frente a los profesionales": the toss height, the racket drop and the arm at contact, each on a bar with the range of 9 of 10 professional serves and their median, with what the bar measures in plain words; the numbers show when the pointer is over the bar,
- "Cómo decide el modelo del brazo que lanza", folded: the confidence (high when every seed agrees, MediaPipe saw the arms and the probability is 0.15 or more from the threshold; low when seeds disagree or under 85 % of the points were seen), the probability against the threshold and the spread over seeds, the model's rates on the test videos and the occlusion map, joints by time, with the joint and stretch that move the answer most,
- from the second attempt on, which points no longer come up, which stay and which are new since the previous attempt.

"Terminar sesión" summarizes the session from the attempts alone (in how many of them each point came up) and gives the drill of the point that came up most for the next practice. With a player ID, the session is added to that player's history in `data/history/`. "Empezar de nuevo" clears the video, the answers and the session for the next player.

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
uv run python -m eval.body                                    # the "Cuerpo" answers against pose measures, and the retest
uv run python -m strokes.serve.body                           # M3, one network for the aspects of the body (--travel: M3b, --balanced: M3c, --parts: M4)
uv run python -m eval.body --checks --model m3b               # a body model (m3, m3b, m3c, m4) aspect by aspect, against the lines
uv run python -m eval.body --ablation                         # M3c's two changes to M3b, one at a time
uv run python -m eval.body --levels                           # what the app says about each aspect, and how sure: points.csv
uv run python -m eval.body --recommend                        # the app's list of points against the answers and the baselines
uv run python -m eval.body --app                              # the app itself on the video of every scored serve: app-points.csv (about 15 minutes)
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

**The body: M3 and M3b.** The labeling tool's "Cuerpo" tab asks about six more aspects of the serve (legs, shoulders, the racket at the peak, the racket drop, the arm at contact, the landing). Ernesto answered them on 167 serves; Hugo could not, so there is no second rater. Each answer was checked against a pose measure written beforehand, with no training (`strokes/serve/body.py`, `eval/body.py`, `experiments/serve/body.csv`; AUROC of the measure for "needs work", on all serves and within EA and within JC):

| Aspect | Needs work | All | Within EA | Within JC | Amateurs against pros | Retest kappa |
|---|---|---|---|---|---|---|
| Legs: the hips go down before the jump | 32 of 166 | 0.90 | 0.60 | 1.00 | 0.68 | 0.67 |
| Shoulders: the tossing one above at the trophy | 81 of 165 | 0.67 | 0.55 | 0.64 | 0.78 | 0.04 |
| Racket at the peak: time from its lowest point to contact | 92 of 167 | 0.73 | 0.33 | 0.39 | 0.89 | 0.77 |
| Racket drop: how low the racket wrist goes | 56 of 166 | 0.55 | 0.26 | 0.43 | 0.65 | 0.70 |
| Arm at contact: the racket wrist above its shoulder | 133 of 166 | 0.99 | – | – | 0.99 | 1.00 |
| Landing: the hips move into the court | 55 of 166 | 0.83 | 0.91 | 0.78 | 0.45 | 0.54 |

Legs, landing and, less clearly, shoulders follow the serve within each player. For the two racket aspects, the answer and the measure disagree within each player (AUROC under 0.5), so the answer follows who serves. Arm at contact is "up" only on professional serves, so it separates amateurs from professionals and nothing else. The racket measures first searched 0.6 s before contact and saturated on every serve; the search was changed to between the trophy and contact before any model was trained. A blind retest of 30 answered serves ("Cuerpo · repetir", same seed, no file names) measures the annotator against themself, on the needs-work answer (`body.csv` also has it on the raw answer). Ernesto answered it on Nov 11, after the first pass of Nov 9 to 11. By the rule fixed beforehand, an aspect with a retest kappa under 0.40 is not trained on or shown: shoulders (0.04) is dropped, the other five pass, from 0.54 (landing) to 1.00 (arm at contact, where nearly every amateur serve needs work).

M3 is M1e's network with one output per aspect the retest keeps (`train.network(outputs=5)`), trained on the 17 joints mirrored so the tossing arm is always the left one and measured from each frame's hips, with a masked loss so unanswered aspects do not count, 85 epochs and 5 seeds, nothing tuned. Folds keep each session on one side, thresholds come from validation (best F1). An aspect is adopted, that is shown as the model's answer, only if it passes every line fixed before training: retest kappa 0.40 or more; validation AUROC within EA and within JC of 0.70 or more and no lower than its measure's; AUROC 0.65 or more on a player left out of training (EA, then JC); at most 20 % of the professionals' serves called "needs work"; and moving the aspect's own body part toward "needs work" must raise the probability on 80 % of the serves answered "as it should be", while moving an unrelated part moves it less than 0.05. M3 fails that last check on every aspect but legs (landing 9 %), and legs fails within EA (0.35) and on both players left out (0.48, 0.46): measuring every frame from its own hips hides how far the body moves. M3b, the same network on positions measured from the hips at contact, was added after seeing that (post hoc, so it is not a test of the original design) and scored by the same lines (`experiments/serve/m3-checks.csv`, `m3b-checks.csv`):

| M3b | Within EA | Within JC | New player EA | New player JC | Pros called needs work | Own edit raises | Unrelated edit moves |
|---|---|---|---|---|---|---|---|
| Legs | 0.39 | 0.98 | 0.65 | 0.97 | 0 % | 100 % | 0.035 |
| Racket at the peak | 0.53 | 0.34 | 0.59 | 0.42 | 24 % | 50 % | 0.006 |
| Racket drop | 0.62 | 0.39 | 0.34 | 0.48 | 7 % | 99 % | 0.057 |
| Arm at contact | – | – | – | – | 34 % | 0 % | 0.052 |
| Landing | 0.913 | 0.780 | 0.89 | 0.72 | 7 % | 98 % | 0.047 |

Shoulders is not trained on (retest kappa 0.04).

No aspect is adopted, by either model. Landing and legs come closest: both carry over to a new player, rarely flag professionals and rest on the right part of the body. Landing beats its untrained measure within EA (0.913 against 0.908) but is below it within JC (0.780 against 0.790), so the network adds little the measure does not already give; legs ranks EA's serves worse than chance (0.39). M3b also calls most THETIS serves, filmed from the front, "needs work" (88 to 100 % on every aspect but the racket drop), so the camera check would still be needed. Ernesto answered the 13 fresh clips of Nov 9 (10 of pro05, the 2 amateur serves of yt2-a and yt3-s07, filmed from the front) "as it should be" on every aspect that could be seen, so check 8 can only count false alarms there: M3b calls "needs work" on 8 % (legs) to 46 % (arm at contact) of the 13, M3 on 0 % (racket drop) to 31 % (arm at contact). Two of Ernesto's own pro05 notes in the toss labels say "poca flexión de rodillas" while the legs answer is "as it should be", one more sign that the answers follow who serves. On the test split, landing is the one aspect with both answers (test AUROC M3b 0.88, M3 0.77).

**More than one point to improve (Nov 12).** The app only spoke of the tossing arm: with one title and one drill built from M1e's answer (`app/api.py`), a serve whose arm is fine was told "no clear problem" and to repeat the same serve. That is 53 of EA's 54 serves and 15 of JC's 44 after the feedback, and on every one of them the annotator marked two or more problems of the body (2.8 on average). The reason the body model was not shown is the one above: the answers follow who serves. "Needs work" is marked on 0 of JC's 37 serves before the feedback and 22 of EA's 54 for the legs, 75 of JC's 81 and 17 of EA's 54 for the racket at the peak, 7 of 81 and 49 of 54 for the racket drop, all but 2 amateur serves for the arm at contact, and 47 of 81 and 8 of 54 for the landing; never on a professional's. Only the legs and the landing change within one player, and a network that sees 17 joints and whose loss is balanced by class alone scores well by telling the players apart.

Before training again, the instants the measures read were audited on stick figures of 12 serves and, for two of EA's, on the frames (kept local). Two things were wrong and each was corrected once, from the question asked and not from the answers (`body.AUDITED`, `experiments/serve/body.csv` keeps both versions). First, `dataset.contact` finds contact about 0.1 s late when the tossing arm is still up while the racket goes up: the racket wrist is highest before the frame found on 53 of EA's 54 serves (a median of 3 frames), away from it on 15 of JC's 81 (most by one frame) and on no professional's, and in the frames the ball is hit at that highest point. The body measures and the new models read contact there (`body.oriented(at_reach=True)`); M1e is left as it was, and for it a late contact only makes the arm seem to drop earlier, so its 2 % of alarms on EA stands. Second, the racket wrist of JC and EA never dips after the trophy, so "its lowest point" was the trophy frame itself, where the racket is still coming up. The racket drop is now how far that wrist falls between the trophy and contact: 0.21 torso lengths for the professionals (median of their 29 serves), 0.07 and 0.02 for JC before and after, 0.00 for EA. It follows the answers with AUROC 0.91 where the first measure had 0.55, and separates amateurs from professionals with 0.97; within a player it still says nothing (0.50 and 0.58). The time of that lowest point stood for the racket at the ball's peak, which the pose cannot show without the ball: that aspect is left with no measure.

Two models were then trained against the shortcut, with everything else as M3b (85 epochs, 5 seeds, session folds, nothing tuned). M3c is M3b with contact at the reach and a loss where every (player, answer) cell of an aspect weighs the same (`body.cell_weights`), so telling the players apart no longer lowers it. M4 is M3c with one network per part of the body, each seeing only that part: hips to feet for the legs and the landing, the height of each elbow and wrist above its shoulder for the three racket aspects (`body.HEADS`, the lesson of M1e). Scored by the same checks (`m3c-checks.csv`, `m4-checks.csv`):

| | Legs within EA | Legs new player EA, JC | Legs precision | Landing within EA, JC | Landing new player EA, JC | Landing precision |
|---|---|---|---|---|---|---|
| M3b | 0.39 | 0.65, 0.97 | 0.44 | 0.91, 0.78 | 0.89, 0.72 | 0.73 |
| M3c | 0.53 | 0.58, 1.00 | 0.67 | 0.90, 0.80 | 0.62, 0.60 | 0.84 |
| M4 | 0.62 | 0.59, 0.97 | 0.70 | 0.90, 0.78 | 0.58, 0.62 | 0.67 |

Precision is the share of the amateur serves a model calls "needs work" on validation that the annotator marked. Both changes do what they were written for within a player: the legs go from ranking EA's serves worse than chance to 0.62, with fewer and better alarms, and M4 calls "needs work" on no professional serve for four of its five aspects. But neither model carries over to a player it never saw as M3b does, and no racket aspect passes that line in any model (the nearest, M4's racket at the peak, has 0.69 on EA and 0.51 on JC). `eval/body.py --ablation` takes M3c's two changes one at a time (`m3c-ablation.csv`): the loss by player alone lifts the legs within EA (0.39 to 0.57) and costs the landing little; contact at the reach alone takes the landing on EA as a new player from 0.89 to 0.68 and the legs from 0.65 to 0.50. So part of how well M3b ranked EA's serves came from where their windows were cut, and with 8 of EA's serves marked for the landing these numbers move by 0.1 or more by chance. The limit is the data, two amateurs whose answers mostly differ between them, not the network.

So the app now says how sure it is instead of saying nothing (`eval/body.py --levels`, `experiments/serve/points.csv`). The first three levels were written on Nov 12, after M3 and M3b's checks and before M3c and M4 were trained; the five lines above are unchanged and still name what is validated. The last two were added the same day after M3c and M4's checks, on the team's decision, so they are post hoc: they relax no line, they name the lines a point shown under them fails.

| Level | What it takes | Today |
|---|---|---|
| Validated model | every line fixed before M3 was trained | nothing |
| Model on trial | retest kappa 0.40; AUROC 0.65 on each player left out; at most 20 % of the professionals; the counterfactual; AUROC 0.70 within at least one player, not against the measure; precision 0.70 among amateurs on validation | the landing (M3b: precision 0.73, 7 % of the professionals) |
| Measure against the professionals | retest kappa 0.40; amateurs against professionals AUROC 0.80; past the 90th percentile of the professionals trained on, at most 20 % of those held out (pro01, pro05) and 70 % of the amateur serves there marked; no contradiction with the answers within a player | the arm at contact (98 % marked, 8 % of the held-out professionals) |
| Exploratory model | the lines of the trial level but the player left out | the legs (M4: precision 0.70, no professional; AUROC 0.59 with EA left out, where the line asks 0.65) |
| Model chosen by value | none of these lines: chosen on Nov 8 by its value on validation (`eval/value.py`) | the tossing arm (M1e: precision 0.20, no professional) |

The first version of this section called the tossing arm validated. It never was: M1e was chosen on Nov 8 by its value on validation, before these lines existed, and had not been scored by them. `eval/body.py` now scores it as a body model on the same validation serves (`toss_point`): of the 50 amateur serves it calls "drops early", 10 were labeled so, a precision of 0.20 where the trial level asks 0.70. The other 40 are 10 of JC's 21 serves before the feedback, 29 of his 44 after it, none of which has the fault by the labels, and 1 of EA's. On the test split, JC's serves of Nov 1, it is right on 13 of 16. It flags no professional serve, and no second player has the fault, so the line of the player left out cannot be scored. The app keeps the point, shows it as "chosen by value" with that precision, and no longer puts it first.

The comparison with the measure was dropped for the trial level because the measure never had to pass the other lines: the landing measure puts 50 % of the held-out professionals past the range, M3b calls "needs work" on 7 % of all of them. The audited racket drop separates amateurs from professionals but only 49 % of the amateur serves past the range are marked (JC's fall of 0.02 to 0.07 is outside it and the annotator calls his drop good), so it gets no point and no drill: the app shows it among the parts it looked at, inside or outside the professionals' range, with that 49 % next to it. As a point by its measure it would be right on 44 of the 48 times it is said on EA's validation serves and on 5 of 49 on JC's, and it would flag 2 of the 17 professionals (the what-if row below); the team left it as a comparison. The racket at the peak and the shoulders are not evaluated, and the page says so.

The app (`app/service.py`, `points`) gives up to three points, the one that was right most often on validation first (the precision of `points.csv`) and between two alike the furthest past its threshold, each with what was measured, that precision, its level, its rates and its drill (`knee_load`, `high_reach` and `land_and_hold` were already in `drills.yaml`). The order was by level at first; it is by precision since the tossing arm has no level of its own. A point read from the legs is left out when MediaPipe saw under 70 % of the leg points, and the landing when the video ends before it. The parts with no point are listed after the points (`others`): fine, inside or outside the professionals' range, not seen, or past their threshold when three surer points came first. With nothing past its threshold the title is "no clear problem". Scored against the answers and the toss labels (`eval/body.py --recommend`, `recommend.csv`; a hit per right point, a false alarm per wrong one, a miss per marked problem with no point, weights of `configs/value.yaml`):

| List (validation, 136 serves) | Precision | Coverage | Told no problem with 2 or more marked | Pros told no problem | Q per amateur analysis |
|---|---|---|---|---|---|
| M1e alone, the app before | 0.20 | 0.03 | 69 of 117 | 100 % | −141.91 |
| The app: M1e, the legs, the landing, the arm at contact | 0.76 | 0.56 | 0 of 117 | 82 % | 41.18 |
| The same, EA and JC by models that never saw them | 0.69 | 0.52 | 1 of 117 | 82 % | 12.50 |
| What if: the app and the racket drop by its measure, not shown | 0.72 | 0.70 | 0 of 117 | 71 % | 72.43 |
| M1e and every M3b aspect, not shown | 0.79 | 0.83 | 0 of 117 | 41 % | 138.97 |
| M1e and every M4 aspect, not shown | 0.74 | 0.76 | 0 of 117 | 94 % | 93.38 |
| Pose measures alone, thresholds on validation | 0.82 | 0.66 | 0 of 117 | 59 % | 89.71 |
| toss_hold always | 0.07 | 0.03 | 0 of 117 | 0 % | −192.65 |

On the test split (JC's 16 serves of Nov 1 and 2 of pro01) the app's list has precision 0.86 and coverage 0.69 (Q133.98 against −60.16 for M1e alone) and gives one of the two professional serves a point. On the 13 clips of Nov 9 it gives 5 points, one to the ten professionals of pro05. By player on validation: 80 points on EA's 54 serves with precision 0.86, and 166 on JC's 65 with 0.72. Point by point (`recommend-points.csv`; "marked" counts the annotator's "needs work", or the toss label, on the 119 amateur serves; "said" counts all 136):

| Point (validation) | Marked | Said | Right | Wrong | Said on the 17 professionals |
|---|---|---|---|---|---|
| Arm at contact (measure) | 117 | 120 | 116 | 4 | 2 |
| Landing (M3b, on trial) | 44 | 57 | 41 | 16 | 1 |
| Legs (M4, exploratory) | 32 | 30 | 21 | 9 | 0 |
| Tossing arm (M1e, by value) | 10 | 42 | 10 | 32 | 0 |
| Racket drop | 56 | 0 | | | |
| Racket at the peak | 77 | 0 | | | |

The arm at contact is marked on all but 2 amateur serves, so it is nearly always right and tells little apart. The landing is the one point that follows the serve within both players. The legs are said on 10 of JC's serves (9 marked) and on 20 of EA's (12 marked); with each player scored by the model that never saw them, on 30 of EA's (14 marked) and on none of JC's 10 marked, which is why the level is exploratory. The tossing arm is said 42 times and not 50 because 8 of its alarms fall on serves with three surer points. With the legs the list covers 0.56 of the marked problems where it covered 0.50, at about the same precision (0.76, 0.74 before). What it still never says is the racket, 133 of the marks. Every model's full list is worth more on these answers than the app's, as before: it covers the racket by repeating what was marked for each player. On EA the app now says the arm at contact on nearly every serve, the legs on 20 of 54 and the racket drop as a number (0.00 against 0.21); what it cannot yet do is tell an amateur with a good racket drop from one without, because the data have no such pair.

Those lists are scored from the windows of the dataset. `eval/body.py --app` runs the app itself on the video of each of the 167 serves (`app.service.run_analysis`: MediaPipe again, the contact it finds, its checks of the recording and its points; `app-points.csv`), to check that the app says what was scored. It asks for a new recording on 4: 3 of EA's and yt3-s07, the clip filmed from the front. On the others it says each point within three serves of how often it was scored. On validation: 243 points with precision 0.76 (the arm at contact 117 times, right on 113; the landing 56, right on 40; the tossing arm 42, right on 10; the legs 28, right on 22), and no point on 15 of the 17 professional serves, the other two getting the arm at contact. On the test split each point is said and right as often as scored. On the 13 clips of Nov 9 it gives 2 points, one to a professional of pro05 and one to an amateur of yt2-a.

What would move an aspect up a level: amateurs with and without each problem recorded the same way, a second rater who does not see the player, and for the racket a camera or model that sees it. None of that arrives before the report, so the limits stand as declared: two amateurs, one rater for the body, labels of the Nov 7 clips that were not blind, and two levels written after seeing M3c and M4. On jc-s30 (test split, Nov 1) the app gives the arm at contact (measure: 0.77 torso lengths against the professionals' median of 1.02), the landing (on trial: the hips move 0.02 torso lengths after contact, the professionals' median is 0.46) and the tossing arm (chosen by value); the legs are fine and the racket drop is inside the professionals' range (0.11, where 9 of 10 professionals reach 0.08).

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
