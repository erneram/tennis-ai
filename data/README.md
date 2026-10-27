# Data

Videos of players are personal data. They are never committed to Git.

| Folder | Content | In Git |
| --- | --- | --- |
| `raw/` | Original serve videos, recorded as described in [docs/grabacion.md](../docs/grabacion.md). | No |
| `processed/` | Derived files: extracted frames, keypoints, overlays. Can be regenerated from `raw/`. | No |
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
| `phase` | Phase with the main opportunity for improvement: `stance`, `toss`, `trophy`, `leg_drive`, `contact`, `follow_through`. Empty when `valid` is `0`. |
| `category` | A category from `strokes/serve/criteria.yaml`. Empty when `valid` is `0`. |
| `drill_id` | The drill for the chosen category, from `strokes/serve/drills.yaml`. Optional. |
| `notes` | Free text from the annotator. |

## What the labels are for

In Spanish, "entrenar" means two different things. In this repository, *training* always means adjusting the model's weights. The player *practices*, and a *coach* teaches the player.

In Project 3 the model is not trained: it is used as downloaded, with a prompt. The labels are only the answer key. They are written once, before looking at the system's answers; the model answers without seeing them, and `eval/run_eval.py` compares its answers with them. The app then gives feedback on new videos that nobody labeled.

A label is the main error of the serve, chosen from the categories in `strokes/serve/criteria.yaml` (for example `toss_arm_bent`). Style is not an error: a personal start position, for example, is never labeled as one.

## Labeling guide

There is no coach, so Ernesto and Hugo label every evaluation serve. Neither is a tennis expert; the written report says so.

1. Each annotator labels alone, in `eval/serve/labels_<name>.csv` (same columns), without looking at the system's answers for those videos.
2. Watch each video at normal speed and in slow motion. Judge it only with the category descriptions and the `observe` notes in `strokes/serve/criteria.yaml`.
3. Set `valid` to `0` when a person could not judge the serve either: player out of frame, serve cut off, too dark, or not a serve.
4. Choose the one problem you see most clearly. If two are equally clear, choose the earlier phase. If you see no clear problem, choose the closest category and write `sin problema claro` in `notes`.
5. Compare the two files with `uv run python -m eval.run_eval --agreement <file A> <file B>`. It prints Cohen's kappa per column and the videos you disagree on.
6. Settle each disagreement together, write the agreed row in `eval/serve/labels.csv`, and report the kappa values.
