from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelRuntimeProfile:
    provider: str
    model_name: str
    role: str
    gpu_ram_gb: float
    num_ctx: int
    max_output_tokens: int
    prompt_budget_tokens: int
    data_profile_budget_tokens: int
    rag_budget_tokens: int
    validation_error_budget_tokens: int
    safety_margin_tokens: int
    compression_enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "role": self.role,
            "gpu_ram_gb": self.gpu_ram_gb,
            "num_ctx": self.num_ctx,
            "max_output_tokens": self.max_output_tokens,
            "prompt_budget_tokens": self.prompt_budget_tokens,
            "data_profile_budget_tokens": self.data_profile_budget_tokens,
            "rag_budget_tokens": self.rag_budget_tokens,
            "validation_error_budget_tokens": self.validation_error_budget_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "compression_enabled": self.compression_enabled,
        }


def resolve_model_runtime_profile(
    *,
    provider: str,
    model_name: str,
    role: str,
    gpu_ram_gb: float = 8.0,
    explicit_num_ctx: int | None = None,
    explicit_max_output_tokens: int | None = None,
    explicit_prompt_budget_tokens: int | None = None,
) -> ModelRuntimeProfile:
    clean_provider = provider.lower().strip()
    clean_model = model_name.strip()
    clean_role = role.lower().strip() or "reasoning"
    gpu_ram = max(1.0, float(gpu_ram_gb or 8.0))
    num_ctx = explicit_num_ctx or _default_num_ctx(clean_provider, clean_model, clean_role, gpu_ram)
    max_output_tokens = explicit_max_output_tokens or _default_output_tokens(clean_role, num_ctx)
    safety_margin = min(512, max(192, int(num_ctx * 0.08)))
    prompt_budget = explicit_prompt_budget_tokens or max(1024, num_ctx - max_output_tokens - safety_margin)
    data_profile_budget = _section_budget(prompt_budget, clean_role, "data_profile")
    rag_budget = _section_budget(prompt_budget, clean_role, "rag")
    validation_budget = _section_budget(prompt_budget, clean_role, "validation")
    return ModelRuntimeProfile(
        provider=clean_provider,
        model_name=clean_model,
        role=clean_role,
        gpu_ram_gb=gpu_ram,
        num_ctx=num_ctx,
        max_output_tokens=max_output_tokens,
        prompt_budget_tokens=prompt_budget,
        data_profile_budget_tokens=data_profile_budget,
        rag_budget_tokens=rag_budget,
        validation_error_budget_tokens=validation_budget,
        safety_margin_tokens=safety_margin,
    )


def _default_num_ctx(provider: str, model_name: str, role: str, gpu_ram_gb: float) -> int:
    if provider != "ollama":
        return 16384
    if role == "vlm":
        return 4096 if gpu_ram_gb < 12 else 8192
    size_b = _model_size_b(model_name)
    if gpu_ram_gb < 10:
        if size_b <= 4.5:
            return 8192
        return 4096
    if gpu_ram_gb < 16:
        if size_b <= 4.5:
            return 12288
        if size_b <= 8.5:
            return 8192
        return 4096
    if gpu_ram_gb < 24:
        if size_b <= 8.5:
            return 16384
        return 8192
    return 32768


def _model_size_b(model_name: str) -> float:
    text = model_name.lower()
    qat_match = re.search(r"e(\d+(?:\.\d+)?)b", text)
    if qat_match:
        return float(qat_match.group(1))
    match = re.search(r"(?<![a-z])(\d+(?:\.\d+)?)b(?![a-z])", text)
    if match:
        return float(match.group(1))
    if "4b" in text:
        return 4.0
    if "7b" in text or "8b" in text:
        return 8.0
    return 7.0


def _default_output_tokens(role: str, num_ctx: int) -> int:
    base = {"spec": 1024, "reasoning": 768, "vlm": 768}.get(role, 768)
    return min(base, max(384, int(num_ctx * 0.25)))


def _section_budget(prompt_budget: int, role: str, section: str) -> int:
    ratios = {
        "reasoning": {"data_profile": 0.35, "rag": 0.12, "validation": 0.10},
        "spec": {"data_profile": 0.30, "rag": 0.18, "validation": 0.14},
        "vlm": {"data_profile": 0.08, "rag": 0.08, "validation": 0.08},
    }
    ratio = ratios.get(role, ratios["reasoning"]).get(section, 0.10)
    return max(256, int(math.floor(prompt_budget * ratio)))


def attach_runtime_profile(model: object, profile: ModelRuntimeProfile) -> object:
    try:
        setattr(model, "_virage_runtime_profile", profile)
    except (AttributeError, TypeError, ValueError):
        try:
            object.__setattr__(model, "_virage_runtime_profile", profile)
        except (AttributeError, TypeError, ValueError):
            pass
    return model


def runtime_profile_from_model(model: object) -> ModelRuntimeProfile | None:
    profile = getattr(model, "_virage_runtime_profile", None)
    return profile if isinstance(profile, ModelRuntimeProfile) else None
