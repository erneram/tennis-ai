# Third-party licenses

Every model, library, dataset, and repository is checked here before use. All reused resources are cited in the written report.

Checked on 2026-10-19 against the GitHub, Hugging Face, and PyPI metadata. Re-check before adding a new resource.

## Libraries and models in use

| Resource | License | Status |
| --- | --- | --- |
| [MLX](https://github.com/ml-explore/mlx) / [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) | MIT | OK. |
| [OpenCV](https://github.com/opencv/opencv-python) | Apache-2.0 | OK. Video reading, frame sampling, and the slow-motion video. |
| [Gradio](https://github.com/gradio-app/gradio) | Apache-2.0 | OK. Web interface; analytics disabled in `app/`. |
| [Transformers](https://github.com/huggingface/transformers) (installed by MLX-VLM) | Apache-2.0 | OK. |
| [llguidance](https://github.com/guidance-ai/llguidance) (installed by MLX-VLM) | MIT | OK. Constrains the model output to the JSON schema. |
| [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) ([MLX 4-bit](https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit)) | Apache-2.0 | OK. Primary vision-language model. |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) ([MLX 4-bit](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-4bit)) | Apache-2.0 | OK. Baseline: smaller model. |

## Candidate models for later iterations

| Model | License | Note |
| --- | --- | --- |
| Qwen3.5-4B | Apache-2.0 | Natively multimodal. |
| Qwen3-VL-8B-Instruct | Apache-2.0 | About 6 GB at 4-bit; fits in 24 GB. |
| Gemma 4 E4B | Apache-2.0 | No 4-bit MLX port found yet. |

## Models to avoid

| Model | Reason |
| --- | --- |
| Qwen2.5-VL-3B | Qwen Research License: non-commercial only. |
| Gemma 3 | Gemma Terms of Use with a prohibited-use policy; gated download. |
| Llama 3.2 Vision | Llama Community License; multimodal models restricted in the EU. |

## Datasets

| Dataset | License | Status |
| --- | --- | --- |
| [Tennis Player Actions Dataset](https://data.mendeley.com/datasets/nv3rpsxhhk/1) | CC BY 4.0 | OK with attribution. |
| [THETIS](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W08/papers/Gourgari_THETIS_Three_Dimensional_2013_CVPR_paper.pdf) | No explicit license found | Check the terms with the authors before use. Do not redistribute. |
| [Roboflow tennis-pose-detection](https://universe.roboflow.com/degree-jpabx/tennis-pose-detection) | Not verified | Check the license on the dataset page before use. |
| [RacketVision](https://github.com/OrcustD/RacketVision/) | MIT (code) | OK. Future work. |
| Own recordings | Player consent | Written consent required. Never committed (see `data/README.md`). |

## Published research behind the serve criteria

Only facts and numbers are used, cited in `strokes/serve/criteria.yaml`; no text or figures are copied.

| Source | Access | Used for |
| --- | --- | --- |
| Kovacs & Ellenbecker (2011), *Sports Health* 3(6):504-513, [doi:10.1177/1941738111414175](https://doi.org/10.1177/1941738111414175) | Free on PubMed Central | Serve stages, common errors (tossing arm dropping early, front knee bent more than 15°, contact high with the elbow slightly bent), start position as personal style. |
| Jacquier-Bret & Gorce (2024), *Frontiers in Sports and Active Living*, [doi:10.3389/fspor.2024.1432030](https://doi.org/10.3389/fspor.2024.1432030) | CC BY | Meta-analysis: knee flexion at the trophy position, shoulder and elbow angles at contact. |
| Tubez et al. (2021), *Sports Biomechanics*, as cited by Gorce & Jacquier-Bret (2024), [doi:10.3389/fspor.2024.1451174](https://doi.org/10.3389/fspor.2024.1451174) | CC BY (citing article) | Shoulder abduction at the trophy position (elbow normally below shoulder height). |

## Reference repositories

| Repository | License | Status |
| --- | --- | --- |
| [SwingForge](https://github.com/SinbadSails/SwingForge) | None | Read for ideas only. Do not copy code. |
| [tennis_analysis (abdullahtarek)](https://github.com/abdullahtarek/tennis_analysis) | None | Read for ideas only. Do not copy code. |
| [Tennis Serve Analysis](https://science.ecosyste.ms/projects/25005) | Not verified | Check before reusing anything. |
