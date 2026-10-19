# Data

Videos of players are personal data. They are never committed to Git.

| Folder | Content | In Git |
| --- | --- | --- |
| `raw/` | Original serve videos, recorded from the defined side angle. | No |
| `processed/` | Derived files: extracted frames, keypoints, overlays. Can be regenerated from `raw/`. | No |
| `annotations/` | Small annotation files (phases, coach labels). Format defined in Phase 1. | Yes |

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

`eval/serve/labels.csv`: one row per serve video, filled in by the coach. Development videos can use the same format in `annotations/serve.csv`.

| Column | Values |
| --- | --- |
| `video` | File name inside `raw/`, for example `p03_s02.mp4`. |
| `player` | Player ID from `players.csv`. |
| `valid` | `1` if the video can be analyzed; `0` if the system should ask for a new recording (wrong angle, out of frame, too dark, not a serve). |
| `phase` | Phase with the main opportunity for improvement: `stance`, `toss`, `trophy`, `leg_drive`, `contact`, `follow_through`. Empty when `valid` is `0`. |
| `category` | A category from `strokes/serve/criteria.yaml`. Empty when `valid` is `0`. |
| `drill_id` | The drill the coach would recommend, from `strokes/serve/drills.yaml`. Optional. |
| `notes` | Free text from the coach. |
