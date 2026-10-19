"""Vision-language model client (MLX-VLM) with JSON-schema-constrained output and validation.

Decoding is constrained by the schema (llguidance), so the model can only produce valid JSON
with allowed values; `validate` double-checks the result before anyone uses it.
"""

import json
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class Generation:
    data: dict
    prompt_tokens: int
    generation_tokens: int
    seconds: float
    peak_memory_gb: float


def validate(data, schema: dict) -> None:
    """Raise TypeError or ValueError if `data` breaks the (flat) object schema."""
    if not isinstance(data, dict):
        raise TypeError("Model output is not a JSON object")
    missing = set(schema["required"]) - set(data)
    if missing:
        raise ValueError(f"Model output misses {sorted(missing)}")
    for key, spec in schema["properties"].items():
        values = data[key] if spec["type"] == "array" else [data[key]]
        allowed = spec.get("enum") or spec.get("items", {}).get("enum")
        if allowed and any(v not in allowed for v in values):
            raise ValueError(f"Model output has a value outside the allowed list for {key!r}")
        if spec["type"] == "string" and not str(data[key]).strip():
            raise ValueError(f"Model output has an empty {key!r}")


class VLM:
    def __init__(self, model_id: str):
        from mlx_vlm import load

        self.model_id = model_id
        self.model, self.processor = load(model_id)

    def generate_json(self, system: str, user: str, images: list[np.ndarray], schema: dict,
                      temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 400,
                      seed: int = 0) -> Generation:  # fmt: skip
        """Generate one JSON object that follows `schema`. Images are BGR arrays."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.structured import build_json_schema_logits_processor

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = apply_chat_template(
            self.processor, self.model.config, messages, num_images=len(images)
        )
        constrain = build_json_schema_logits_processor(self.processor.tokenizer, schema)
        pictures = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]

        start = time.perf_counter()
        result = generate(
            self.model,
            self.processor,
            prompt,
            image=pictures or None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            logits_processors=[constrain],
            verbose=False,
        )
        seconds = time.perf_counter() - start
        if result.finish_reason == "length":
            raise ValueError(f"Model output was cut at max_tokens={max_tokens}")
        data = json.loads(result.text)
        validate(data, schema)
        return Generation(
            data=data,
            prompt_tokens=result.prompt_tokens,
            generation_tokens=result.generation_tokens,
            seconds=seconds,
            peak_memory_gb=result.peak_memory,
        )
