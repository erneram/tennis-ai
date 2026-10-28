# Serve Lab

## Proposal summary

Serve Lab is an interactive assistant specialized in analyzing the **tennis serve**. From a short video, it splits the motion into phases, identifies which phase holds the main opportunity for improvement, explains the observation with the frames that support it, and suggests a drill. It also keeps a history of each session to show the player's progress and suggest the focus of the next practice.

The serve was chosen because it is the only stroke the player fully controls: it does not depend on the opponent's ball, it starts from a static position, and it is executed the same way in any match situation. As a result, what is corrected in practice carries over directly to real play, unlike strokes such as the forehand, which change with the height, speed, and direction of the incoming ball.

The proposal does not aim to build a general-purpose chatbot. Its value lies in supporting the player's progress through visual evidence, criteria taken from published serve biomechanics, and follow-up across sessions.

### Change of approach (October 19, 2026)

The team has no coach. The player records their own serves and uses Serve Lab as their coach. This changes two things:

- **Criteria:** the problem categories and reference ranges come from published serve biomechanics studies, cited in `strokes/serve/criteria.yaml`. In Project 3 every category is judged from the frames.
- **Reference answers:** Ernesto and Hugo label each evaluation serve independently, following the guide in `data/README.md` and without seeing the system's answer. Their agreement (Cohen's kappa) is reported, and they settle disagreements together. The labels come from non-experts, which is a limitation of the evaluation.

### Change of approach (October 27, 2026)

Project 3 must be a language-model product. Pose estimation (MediaPipe Pose, YOLO Pose), the measurements computed from it, and the fixed-rule baseline moved to the Final Project (branch `proyecto-final`). In Project 3, only the vision-language model looks at the video: it finds the phases in evenly spaced frames and judges the key frame of each phase.

## User experience

1. The player records one or more serves from a set side angle.
2. The app shows the video in slow motion and marks each phase.
3. The assistant points out the phase with the main opportunity for improvement and shows the frames that support the observation.
4. The system recommends a short drill, taken from the drill knowledge base.
5. The player uploads a new attempt and receives a visual comparison with the previous one.
6. At the end of the session, the assistant summarizes progress against previous sessions and proposes the focus of the next practice.

The key differentiator is the loop of **practice, get feedback, try again, and follow up**, rather than just measuring an isolated stroke.

## Serve phases analyzed

| Phase | What is observed |
| --- | --- |
| 1. Stance | Foot placement relative to the baseline and body orientation. |
| 2. Ball toss | Height and placement of the toss relative to the body; stability of the tossing arm. |
| 3. Trophy position | Shoulder alignment, racket elbow, and tossing arm extended toward the ball. |
| 4. Leg drive | Knee bend and weight transfer before the push-off. |
| 5. Contact | Arm extension and height of the contact point. |
| 6. Follow-through | Continuity of the motion, balance, and landing. |

The feedback categories are based on published serve biomechanics studies. For the prototype, they will be limited to those that can be reliably observed from a single angle.

## Technical approach

Only the vision-language model looks at the video. The system combines four components:

- **Frame sampling:** OpenCV checks the frame rate and length of the video and takes evenly spaced frames (32 by default). Each frame gets its number drawn on it.
- **Phase location (first model call):** the model decides whether the video can be analyzed and, if not, gives the reason to record again. Otherwise, it picks the frame that best shows each of the six phases.
- **Feedback (second model call):** the model receives the six key frames, what to observe in each phase, the problem categories, and the drills, and writes one prioritized observation with its drill. Both calls return JSON constrained by a schema, so the model can only choose known phases, categories, drills, and frames.
- **History and planning:** stores the results of each session and lets the model generate a progress summary and the focus of the next practice.

There are no measurements, so the model never states numbers: it describes what the frames show. A single-angle video cannot accurately reconstruct the player's full biomechanics. When the evidence is insufficient (wrong angle, player out of frame, poor lighting), the system must ask for a new recording instead of issuing an observation.

## Tools and resources considered

### Vision-language models

| Option | Characteristics | Use in the project |
| --- | --- | --- |
| **Qwen3-VL-4B-Instruct (4-bit, MLX)** | Open model (Apache-2.0) that reads several images in one request; about 3 GB. | Primary model for both calls. |
| **Qwen3-VL-2B-Instruct (4-bit, MLX)** | Same family, half the size. | Baseline: shows how much the larger model adds. |

Pose estimation models (MediaPipe Pose, YOLO Pose) are not used in Project 3. They are part of the Final Project.

### Tennis datasets

Public datasets do not include technical corrections for each serve, so they do not replace our own evaluation set. They are considered as options to add serve examples from unseen players.

| Option | Content | Possible use |
| --- | --- | --- |
| **Tennis Player Actions Dataset** | 2,000 images of four actions (forehand, backhand, ready position, and serve), with body keypoint annotations in COCO format. | Final Project: verify pose accuracy in serve positions. |
| **THETIS** | 1,980 videos of 12 tennis actions, including flat, kick, and slice serves, with beginner to intermediate players. Filmed from the front and without a ball. | Additional source of serves from unseen players; the difference in angle must be taken into account. |
| **Roboflow Universe** | Datasets and models for tennis pose classification (forehand, backhand, ready position, and serve). | Automatically check whether a video contains a serve before analyzing it. |
| RacketVision | Ball and racket pose annotations for tennis, badminton, and table tennis. | Future work: ball and racket tracking. |

### Reference repositories

| Option | Content | Possible use |
| --- | --- | --- |
| **Tennis Serve Analysis** | Web app that analyzes serves from front and side views, detects key moments, and tracks the ball. | Reference for serve phase segmentation. |
| **SwingForge** | Local stroke analysis with MediaPipe, joint angle computation, scoring, and feedback. | Final Project: reference for angle computation. |
| **tennis_analysis (abdullahtarek)** | Player and ball detection with YOLO and court keypoint detection. | Future work: court position analysis during matches. |

Before using any dataset or repository, its license will be checked, and all reused resources will be cited in the written report.

## Use of the Mac M5

The proposal starts from an existing multimodal model, run locally with MLX-VLM. A quantized model of roughly four billion parameters would be an initial candidate, subject to the available unified memory and performance tests.

The model never receives the full video, to control context window usage and latency: it receives 32 small frames (512 px) to find the phases, and then six larger key frames (1,024 px) to judge them. In the first test on the development videos, both calls took about 15 s per serve on the Mac M5 Pro, with a memory peak of about 4.8 GB.

The model is not trained in Project 3: it is used as downloaded, with a prompt. The labeled serves are only the answer key for the evaluation ([what the labels are for](../data/README.md#what-the-labels-are-for)).

## Prototype scope

To complete the project before the November 2, 2026 fair, the prototype is limited to:

- A single stroke: the serve.
- Short videos recorded from a defined side angle.
- Six analyzed phases and a small number of feedback categories based on published biomechanics.
- One main recommendation per attempt.
- Comparison between attempts and a basic history across sessions.

Out of scope: full match analysis, other strokes, ball speed or direction, medical diagnoses, and claims of professional-level accuracy without validation.

## Phased implementation plan

On October 27 the plan changed: the pose work of phases 1 and 2 moved to the Final Project (branch `proyecto-final`), and the tables below show the Project 3 version. The work is split into phases implemented in order, since each one uses the output of the previous one. Within each phase, tasks are divided between the two members by area of the repository, so each person works in different folders and commit conflicts are avoided.

Broadly, **Ernesto** is in charge of the technical core (`core/`): frames, vision-language model, history, and running experiments. **Hugo** is in charge of the serve module (`strokes/serve/`), data, evaluation, and the interface. Both must understand the full pipeline, because during the fair any member may be asked about any part of it.

### Phase 0: Repository setup (October 19)

| Ernesto | Hugo |
| --- | --- |
| Create the repository with the folder structure, the Python environment, and `.gitignore`. | Set up his Git identity, create `.env.example`, and draft the README skeleton. |

**Outcome:** repository ready, with test commits from both members recorded under their own identities.

### Phase 1: Data and frames (October 19–22)

| Ernesto | Hugo |
| --- | --- |
| Implement `core/video/`: video loading and frame extraction. | Record the first serves from the defined side angle and organize them in `data/raw/`. |
| Implement `core/analysis.py`: quality checks and evenly spaced, numbered frames. | Define the phases and feedback categories from published biomechanics. |
| Generate the slow-motion video with the current phase. | Define the serve annotation format in `data/annotations/`. |

**Outcome:** any serve video is checked and turned into frames for the model.

### Phase 2: Phase location with the model (October 22–25)

| Ernesto | Hugo |
| --- | --- |
| Implement `core/llm/`: model with MLX-VLM and JSON output constrained by a schema. | Implement `strokes/base.py`, the common interface for all strokes. |
| Implement the first call: the model picks one frame per phase or asks for a new recording. | Write in `criteria.yaml` how to find each phase and what to observe in it. |
| | Write `criteria.yaml` and `drills.yaml` with criteria from published biomechanics, and the drills. |

**Outcome:** the model splits each serve into phases and asks for a new recording when it cannot.

### Phase 3: Feedback and evaluation (October 25–28)

| Ernesto | Hugo |
| --- | --- |
| Implement the second call: one observation, category, and drill from the key frames. | Complete the evaluation set of 30 to 50 serves, labeled by both members. |
| Design and version the prompts in `strokes/serve/prompts/`. | Implement `eval/run_eval.py` with the quality and operational metrics. |
| Run the iterations (number of frames, prompt, temperature) and log them in `experiments/`. | Run the two baselines: a simple prompt without criteria, and a smaller model (Qwen3-VL-2B). |

**Outcome:** the full system generates feedback, and there are metrics that can be compared against the baselines.

### Phase 4: Final product (October 28–31)

| Ernesto | Hugo |
| --- | --- |
| Implement `core/history/`: session history, progress summary, and practice plan. | Implement the interface in `app/`: video upload, results, evidence frames, and drill. |
| Prompt iterations against the evaluation set. | Implement the visual comparison between attempts. |
| Measure latency and memory usage on the Mac M5. | Error handling: invalid videos, requests for a new recording, and wait times. |

**Outcome:** a product usable end to end by someone outside the group.

### Phase 5: Final evaluation and fair preparation (November 1)

| Ernesto | Hugo |
| --- | --- |
| Run the final evaluation with unseen players and prepare the results tables. | Design the poster with the architecture diagram and the main metrics. |
| Prepare cached responses as a backup in case the live system fails. | Record the 1 to 3 minute demo video. |

Pitch rehearsal together.

**Outcome:** live demo ready, with poster, video, and a backup plan.

### Phase 6: Fair and written report (November 2–6)

| Ernesto | Hugo |
| --- | --- |
| Sections 2.2 (data and pipeline), 2.3 (model and adaptation), 2.4 (evaluation), and 2.5 (iterations). | Sections 2.1 (problem and user), 2.6 (final product), 2.7 (limitations and ethics), and 2.8 (conclusions). |
| Check that the metrics in the README are reproducible. | Complete the README with installation and run instructions. |

Both review the full document before submitting it on November 6.

**Outcome:** fair presented on November 2, and the written report, repository, and video submitted on November 6.

## Evaluation

**Evaluation set:** 30 to 50 recorded serves, including players not used during development. For each serve, Ernesto and Hugo independently indicate whether the video is valid, the phase with the main opportunity for improvement, the problem category, and the appropriate drill, without seeing the system's answer. They settle disagreements together, and the agreed labels are the reference answers. Inter-annotator agreement (Cohen's kappa) is reported.

**Quality metrics:**

- Agreement with the reference labels on the identified phase (accuracy).
- Agreement on the problem category (precision, recall, and F1).
- Faithfulness to the evidence: percentage of observations that the frames do not support (hallucination rate). Both members review each observation next to its frames; the evaluation script also logs an automatic proxy (the category belongs to another phase).
- Clarity and usefulness of the recommendation, rated by the players using a rubric.
- Rate of correct requests for a new recording when the video is not valid.

**Operational metrics:** average latency per analyzed serve, tokens consumed per request, and memory usage on the Mac M5.

**Comparison baselines:**

- The same model with a simple prompt, without criteria or phase guide.
- A smaller model (Qwen3-VL-2B) with the same prompts.

Every change in model, prompt, or parameter will be logged as an iteration with its metrics.

## Commercial potential

A possible commercial application would be a follow-up tool for academies and coaches, letting players practice the serve between lessons and arrive with a record of their progress. It is not intended as a replacement for professional coaching. Local processing reduces the need to send videos to external services. Before going commercial, the licenses of the models and data must be verified, as well as consent to use the recordings.

## Future work

- Full match analysis with a fixed camera: player position on the court, poorly covered zones, and their relationship to lost points.
- Adding other strokes, taking the different game situations into account.
- Ball tracking to measure toss height and serve direction.
- Pose estimation to measure angles and find the phases (Final Project, branch `proyecto-final`).

## Scalable repository structure

Although the prototype is limited to the serve, the repository will be organized so that other strokes (forehand, backhand, volley) and on-court play analysis can be added later without rebuilding the system.

The principle is to separate what **all analyses share** from what **is specific to each stroke**:

- **Shared core:** video loading, frame sampling, calls to the vision-language model, player history, and interface. Written only once.
- **Per-stroke modules:** each stroke has its own folder with its phases, criteria, drills, prompts, and evaluation set. Adding a new stroke means creating a new folder with the same structure.

```
serve-lab/
├── README.md
├── .env.example             # environment variables, no real keys
├── configs/                 # frames, model, and generation parameter settings
├── core/                    # core shared by all analyses
│   ├── video/               # video loading and frame extraction
│   ├── analysis.py          # quality checks, sampled frames, slow-motion video
│   ├── llm/                 # vision-language model client and JSON validation
│   └── history/             # sessions and player history
├── strokes/                 # one module per stroke, all with the same structure
│   ├── base.py              # interface every stroke implements, and the two model calls
│   ├── serve/               # current prototype
│   │   ├── __init__.py      # phases and their names
│   │   ├── criteria.yaml    # criteria with their published sources
│   │   ├── drills.yaml      # recommended drills
│   │   └── prompts/         # versioned prompts
│   └── forehand/            # future: same structure as serve/
├── court/                   # future: court, player, and position detection in matches
├── data/                    # videos and annotations (kept out of Git if large or private)
│   ├── raw/
│   └── annotations/
├── eval/                    # evaluation sets and metric scripts, per stroke
│   ├── serve/
│   └── run_eval.py
├── experiments/             # iteration log (model, prompt, metrics)
├── app/                     # user interface
└── tests/
```

**Rules to keep it scalable:**

- The core must not contain logic specific to any stroke. Anything that depends on the serve lives in `strokes/serve/`.
- All strokes implement the same interface defined in `strokes/base.py`: phases, criteria, drills, and prompts. The app picks the module based on the selected stroke.
- Criteria, drills, and prompts are stored as versioned configuration files, not inside the code, so they can be reviewed and iterations can be compared.
- Each stroke has its own evaluation set in `eval/`, so adding a new one does not affect the metrics of existing ones.
- On-court play analysis will be added in `court/` and will reuse the model client and history from the core.

## Initial references

- [MLX-VLM: inference and fine-tuning of multimodal models on Apple Silicon](https://github.com/Blaizzy/mlx-vlm)
- [Tennis player actions dataset for human pose estimation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11282921/)
- [Tennis Player Actions Dataset on Mendeley Data](https://data.mendeley.com/datasets/nv3rpsxhhk/1)
- [THETIS: Three Dimensional Tennis Shots (CVPR Workshops 2013)](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W08/papers/Gourgari_THETIS_Three_Dimensional_2013_CVPR_paper.pdf)
- [Tennis Pose Detection on Roboflow Universe](https://universe.roboflow.com/degree-jpabx/tennis-pose-detection)
- [RacketVision](https://github.com/OrcustD/RacketVision/)
- [SwingForge](https://github.com/SinbadSails/SwingForge)
- [tennis_analysis (abdullahtarek)](https://github.com/abdullahtarek/tennis_analysis)
- [Tennis Serve Analysis](https://science.ecosyste.ms/projects/25005)
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
