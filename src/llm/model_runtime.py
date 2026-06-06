from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

ModelRole = Literal["reasoning", "spec", "vlm", "embedding"]
DEFAULT_GPU_RAM_GB = 8.0


@dataclass(frozen=True, slots=True)
class ModelRuntimeProfile:
    provider: str
    model_name: str
    role: str
    gpu_ram_gb: float
    num_ctx: int
    max_output_tokens: int
    prompt_budget_tokens: int

    def section_budget(self, fraction: float | str, *, minimum: int = 128) -> int:
        if isinstance(fraction, str):
            ratios = {
                "query_request": 0.74,
                "analysis_planner": 0.72,
                "spec_generation": 0.86,
                "vlm_analysis": 0.72,
                "visual_judge": 0.72,
                "rag_guidance": 0.20,
                "validation_error": 0.14,
                "data_profile": 0.32,
            }
            ratio = ratios.get(fraction, 0.50)
        else:
            ratio = fraction
        value = int(self.prompt_budget_tokens * float(ratio))
        return max(minimum, value)

    def as_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "role": self.role,
            "gpu_ram_gb": self.gpu_ram_gb,
            "num_ctx": self.num_ctx,
            "max_output_tokens": self.max_output_tokens,
            "prompt_budget_tokens": self.prompt_budget_tokens,
        }


def build_model_runtime_profile(
        *,
        provider: str,
        model_name: str,
        role: str,
        gpu_ram_gb: float = DEFAULT_GPU_RAM_GB,
        configured_num_ctx: int | None = None,
        configured_max_output_tokens: int | None = None,
) -> ModelRuntimeProfile:
    provider_name = _clean_provider(provider)
    model = _clean_model_name(model_name)
    ram_gb = max(1.0, float(gpu_ram_gb or DEFAULT_GPU_RAM_GB))
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



def resolve_model_runtime_profile(
        *,
        provider: str,
        model_name: str,
        role: str,
        gpu_ram_gb: float = DEFAULT_GPU_RAM_GB,
        configured_num_ctx: int | None = None,
        configured_max_output_tokens: int | None = None,
) -> ModelRuntimeProfile:
    return build_model_runtime_profile(
        provider=provider,
        model_name=model_name,
        role=role,
        gpu_ram_gb=gpu_ram_gb,
        configured_num_ctx=configured_num_ctx,
        configured_max_output_tokens=configured_max_output_tokens,
    )

def attach_runtime_profile(llm: Any, profile: ModelRuntimeProfile) -> Any:
    object.__setattr__(llm, "_virage_runtime_profile", profile)
    return llm


def runtime_profile_from_model(llm: Any, *, role: str = "reasoning") -> ModelRuntimeProfile:
    attached = getattr(llm, "_virage_runtime_profile", None)
    if isinstance(attached, ModelRuntimeProfile):
        return attached
    provider = _provider_from_model(llm)
    model_name = _model_name_from_model(llm)
    configured_num_ctx = _int_attr(llm, "num_ctx")
    configured_output = _int_attr(llm, "num_predict") or _int_attr(llm, "max_tokens")
    return build_model_runtime_profile(
        provider=provider,
        model_name=model_name,
        role=role,
        gpu_ram_gb=DEFAULT_GPU_RAM_GB,
        configured_num_ctx=configured_num_ctx,
        configured_max_output_tokens=configured_output,
    )


def _clean_provider(provider: str) -> str:
    return provider.lower().strip() or "unknown"


def _clean_model_name(model_name: str) -> str:
    return model_name.strip() or "unknown"


def _provider_from_model(llm: Any) -> str:
    module = type(llm).__module__.lower()
    if "ollama" in module:
        return "ollama"
    if "openai" in module:
        return "openai"
    if "huggingface" in module:
        return "huggingface"
    return type(llm).__name__.lower()


def _model_name_from_model(llm: Any) -> str:
    for attr in ("model", "model_name", "repo_id"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return type(llm).__name__


def _int_attr(obj: Any, name: str) -> int | None:
    value = getattr(obj, name, None)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


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
