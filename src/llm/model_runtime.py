from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ModelRole = Literal["reasoning", "spec", "vlm", "embedding"]


@dataclass(frozen=True, slots=True)
class ModelRuntimeProfile:
    provider: str
    model_name: str
    role: str
    gpu_ram_gb: float
    num_ctx: int
    max_output_tokens: int
    prompt_budget_tokens: int

    def section_budget(self, fraction: float, *, minimum: int = 128) -> int:
        value = int(self.prompt_budget_tokens * fraction)
        return max(minimum, value)


def build_model_runtime_profile(
        *,
        provider: str,
        model_name: str,
        role: str,
        gpu_ram_gb: float = 8.0,
        configured_num_ctx: int | None = None,
        configured_max_output_tokens: int | None = None,
) -> ModelRuntimeProfile:
    provider_name = provider.lower().strip()
    model = model_name.strip()
    ram_gb = max(1.0, float(gpu_ram_gb or 8.0))
    num_ctx = configured_num_ctx or _recommended_num_ctx(provider_name, model, ram_gb, role)
    max_output = configured_max_output_tokens or _recommended_output_tokens(role, num_ctx)
    reserved_system_tokens = 384
    prompt_budget = max(512, int(num_ctx) - int(max_output) - reserved_system_tokens)
    return ModelRuntimeProfile(
        provider=provider_name,
        model_name=model,
        role=role,
        gpu_ram_gb=ram_gb,
        num_ctx=int(num_ctx),
        max_output_tokens=int(max_output),
        prompt_budget_tokens=int(prompt_budget),
    )


def _recommended_num_ctx(provider: str, model_name: str, gpu_ram_gb: float, role: str) -> int:
    if provider != "ollama":
        return 8192 if role != "vlm" else 4096
    size_b = _model_size_b(model_name)
    if gpu_ram_gb <= 6.0:
        return 4096
    if role == "vlm":
        return 4096 if size_b >= 7.0 else 8192
    if size_b <= 4.5:
        return 8192 if gpu_ram_gb >= 8.0 else 4096
    if size_b <= 8.5:
        return 4096 if gpu_ram_gb <= 10.0 else 8192
    return 4096


def _recommended_output_tokens(role: str, num_ctx: int) -> int:
    if role == "spec":
        return min(1536, max(768, num_ctx // 8))
    if role == "reasoning":
        return min(1024, max(512, num_ctx // 10))
    if role == "vlm":
        return min(768, max(384, num_ctx // 12))
    return min(768, max(384, num_ctx // 12))


def _model_size_b(model_name: str) -> float:
    normalized = model_name.lower()
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*b", normalized)
    if match:
        return float(match.group(1))
    match = re.search(r":e(\d+(?:\.\d+)?)b", normalized)
    if match:
        return float(match.group(1))
    if "e2b" in normalized:
        return 2.0
    if "e4b" in normalized:
        return 4.0
    if "4b" in normalized:
        return 4.0
    if "7b" in normalized or "8b" in normalized:
        return 8.0
    if "12b" in normalized:
        return 12.0
    return 7.0
