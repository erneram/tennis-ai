# Data

Videos of players are personal data. They are never committed to Git.

| Folder | Content | In Git |
| --- | --- | --- |
| `raw/` | Original serve videos, recorded as described in [docs/grabacion.md](../docs/grabacion.md). | No |
| `processed/` | Derived files, such as the demo answers in `demo_cache/`. Can be regenerated. | No |
| `demo/` | Videos shown at the fair (`python -m app --demo`). Only players with `consent_image` = `yes`. | No |
| `annotations/` | Small annotation files (players, development labels). | Yes |

## Rules

- Record a player only after they give written consent for the recording to be used in this project.
- Name files by an anonymous player ID, never by the player's name.
- Share raw videos between members outside Git (for example, a shared drive with restricted access).
- Before using a public dataset, check its license in `docs/licenses.md`.

## Annotation format

`annotations/players.csv`: one row per player. No names, only the anonymous ID.

| Column | Values |
| --- | --- |
| `player` | Anonymous ID, for example `p03`. |
| `handedness` | `right` or `left`. |
| `level` | `beginner`, `intermediate`, or `advanced`. |
| `consent_recording` | `yes` if the player signed consent to be recorded. |
| `consent_image` | `yes` if their face may appear in the demo video and poster. |
| `notes` | Free text. |

`eval/serve/labels.csv`: one row per serve video, with the labels both annotators agreed on (see the guide below). Development videos can use the same format in `annotations/serve.csv`.

| Column | Values |
| --- | --- |
| `video` | File name inside `raw/`, for example `p03_s02.mp4`. |
| `player` | Player ID from `players.csv`. |
| `valid` | `1` if the video can be analyzed; `0` if the system should ask for a new recording (wrong angle, out of frame, too dark, not a serve). |
| `phase` | Phase with the main opportunity for improvement: `stance`, `toss`, `trophy`, `leg_drive`, `contact`, `follow_through`, or `none` with `no_clear_problem`. Empty when `valid` is `0`. |
| `category` | A category from `strokes/serve/criteria.yaml`. Empty when `valid` is `0`. |
| `drill_id` | The drill for the chosen category, from `strokes/serve/drills.yaml`. Optional. |
| `notes` | Free text from the annotator. |

## What the labels are for

In Spanish, "entrenar" means two different things. In this repository, *training* always means adjusting the model's weights. The player *practices*, and a *coach* teaches the player.

In Project 3 the model is not trained: it is used as downloaded, with a prompt. The labels are only the answer key. They are written once, before looking at the system's answers; the model answers without seeing them, and `eval/run_eval.py` compares its answers with them. The app then gives feedback on new videos that nobody labeled.

In Project 3 a label was the main error of the serve, chosen from the categories in `strokes/serve/criteria.yaml`. In the Final Project it only says whether the tossing arm drops early (`toss_arm_drops_early`) or not (`no_clear_problem`), see the guide below. Style is not an error: a personal start position, for example, is never labeled as one.

## Labeling guide

There is no coach, so Ernesto and Hugo label every evaluation serve. Neither is a tennis expert; the written report says so.

1. Each annotator labels alone, in `eval/serve/labels_<name>.csv` (same columns), without looking at the system's answers, at `eval/serve/claude_labels.csv`, or at the other annotator's file. `uv run python -m eval.label` (http://127.0.0.1:7861) shows each video at normal speed and in slow motion next to a professional serve (`pro02_s01`) and writes the file; the phase and the drill follow from the answers.
2. The tool asks two yes/no questions that need no tennis knowledge, only what can be seen. Since the Final Project, the label is this one problem, `toss_arm_drops_early` in `strokes/serve/criteria.yaml` (Kovacs 2011), and not the most clear of seven categories: two non-experts cannot diagnose like a coach, and the other categories had 1 to 3 videos each.
3. "Is the player seen from the side and full body, with the hand that tosses the ball visible?" No sets `valid` to `0`: player out of frame, filmed from behind, serve cut off, too dark, or not a serve.
4. "When the racket is lowest, behind the back (just before it goes up to hit), has the hand that tossed the ball already dropped?" Yes, to shoulder height or lower, is `toss_arm_drops_early`; no, still above the shoulder and pointing at the ball, is `no_clear_problem`.
5. Compare the two files with `uv run python -m eval.run_eval --agreement <file A> <file B>`. It prints Cohen's kappa per column and the videos you disagree on.
6. Settle each disagreement together in the tab "Decidir juntos" of the same tool: it shows only the videos you answered differently, with each answer, and writes `eval/serve/labels.csv` (the rows you agree on plus your decisions). Report the kappa values. Clips one annotator labeled alone at the team's request (the YouTube videos of Nov 9, `ONE_RATER` in `eval/label.py`) go in with that annotator's label and the note `one rater: <name>`. When there is nothing to settle, `uv run python -m eval.label --merge` writes `labels.csv` without opening the tool.
7. Only then, in the tab "Revisar respuestas" of the same tool, judge the checks of the two runs of the final Project 3 version (JC, 126 checks, and the professionals, 66): for each key frame, whether what the model says it sees is in the frame, and whether its answer is right. Only videos you already labeled are shown. `uv run python -m eval.run_eval --review eval/serve/runs/<run>.jsonl eval/serve/review_*.csv` prints the hallucination rate (sentences that describe something the frame does not show) and writes it in the run's row of `experiments/iterations.csv`.
8. Optional, for faults other than the tossing arm: the tab "Cuerpo" asks seven multiple-choice questions about the rest of each judged serve (legs, shoulders, where the racket is when the ball is highest, whether it drops behind the back, the arm and the contact point at impact, the landing), each with what to look at and the ideal, and "No se ve" (`unseen`) when it cannot be seen. The questions and answer codes are `ASPECTS` in `eval/label.py`; each annotator's answers go to `eval/serve/aspects_<name>.csv`. The players come in turn (JC before the feedback, JC after it, EA, each professional), so a partial pass still covers all of them. These answers are not labels yet: a fault becomes a label only once both annotators answered it and their agreement is measured, as above.

`eval/serve/claude_labels.csv` holds a third opinion on every video: Claude (a different LLM from the one in the app) judged each JC serve by eye from 20 frames around contact (`uv run python -m eval.contact_sheet <videos>`); the professional serves are marked no clear problem without a frame-by-frame check. It is not a reference answer and must not be read before labeling. After the agreed labels exist, `--agreement eval/serve/labels.csv eval/serve/claude_labels.csv` measures how far an LLM judge agrees with the team, as the brief asks for LLM-as-a-judge (section 2.4).

Agreement on the 55 videos (2026-11-06): Ernesto and Hugo, labeling alone, kappa 1.00 on `valid` and 0.64 on `category` (36 of 48 valid videos the same, the 12 others all JC and settled together); the agreed labels against `claude_labels.csv`, kappa 0.92 on `valid` and 0.59 on `category`.
