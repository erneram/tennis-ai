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
