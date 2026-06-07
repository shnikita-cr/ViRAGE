from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


@dataclass(frozen=True)
class ImageTextCosineResult:
    model_name: str
    cosine: float


@dataclass(frozen=True)
class ImageTextCosineError:
    model_name: str
    error_type: str
    error: str


@dataclass(frozen=True)
class ImageTextCosineBatchResult:
    results: list[ImageTextCosineResult]
    errors: list[ImageTextCosineError] = field(default_factory=list)

    @property
    def mean_cosine(self) -> float:
        if not self.results:
            raise ValueError("Image-text cosine result is empty.")
        return sum(item.cosine for item in self.results) / len(self.results)


class HuggingFaceImageTextCosineModel:
    def __init__(self, *, model_name: str, device: str, dtype: torch.dtype, trust_remote_code: bool = False) -> None:
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype, trust_remote_code=trust_remote_code).to(self.device)
        self.model.eval()

    def score(self, *, image_path: str | Path, text: str) -> ImageTextCosineResult:
        if not text.strip():
            raise ValueError("Image-text cosine requires non-empty text.")
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")
        with Image.open(image_file) as image:
            rgb_image = image.convert("RGB")
            inputs = self.processor(text=[text], images=[rgb_image], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_features, text_features = self._features(outputs)
        cosine = torch.nn.functional.cosine_similarity(image_features, text_features).reshape(-1)[0].item()
        return ImageTextCosineResult(model_name=self.model_name, cosine=float(cosine))

    def _features(self, outputs) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(outputs, "image_embeds") and hasattr(outputs, "text_embeds"):
            return outputs.image_embeds, outputs.text_embeds
        if hasattr(outputs, "pooler_output"):
            raise RuntimeError(f"Model {self.model_name!r} did not return separate image/text embeddings.")
        raise RuntimeError(f"Model {self.model_name!r} is not compatible with generic image-text cosine extraction.")


class ImageTextCosineEvaluator:
    def __init__(self, *, model_names: Sequence[str], device: str = "cuda", dtype: str = "float16") -> None:
        if not model_names:
            raise ValueError("At least one image-text model must be configured.")
        torch_dtype = self._dtype(dtype)
        self.model_errors: list[ImageTextCosineError] = []
        self.models: list[HuggingFaceImageTextCosineModel] = []
        for model_name in model_names:
            try:
                self.models.append(
                    HuggingFaceImageTextCosineModel(
                        model_name=model_name,
                        device=device,
                        dtype=torch_dtype,
                        trust_remote_code=self._requires_remote_code(model_name),
                    )
                )
            except (RuntimeError, ValueError, OSError, ImportError, FileNotFoundError) as exc:
                self.model_errors.append(
                    ImageTextCosineError(
                        model_name=model_name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                )

    def score(self, *, image_path: str | Path, task_text: str) -> ImageTextCosineBatchResult:
        results: list[ImageTextCosineResult] = []
        errors = list(self.model_errors)
        for model in self.models:
            try:
                results.append(model.score(image_path=image_path, text=task_text))
            except (RuntimeError, ValueError, OSError, FileNotFoundError) as exc:
                errors.append(
                    ImageTextCosineError(
                        model_name=model.model_name,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                )
        return ImageTextCosineBatchResult(results=results, errors=errors)

    @staticmethod
    def _dtype(value: str) -> torch.dtype:
        normalized = value.strip().lower()
        if normalized == "float16":
            return torch.float16
        if normalized == "bfloat16":
            return torch.bfloat16
        if normalized == "float32":
            return torch.float32
        raise ValueError(f"Unsupported image-text dtype: {value!r}.")

    @staticmethod
    def _requires_remote_code(model_name: str) -> bool:
        return model_name.lower().startswith("jinaai/")
