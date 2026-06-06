from __future__ import annotations

from src.application.config.project_config import ModelRoleConfig
from src.application.config.settings import ViRAGESettings
from src.llm.model_runtime import resolve_model_runtime_profile
from src.llm.prompt_budget import PromptSection, build_budgeted_prompt, estimate_tokens


def test_local_4b_profile_uses_larger_context_on_default_8gb_gpu() -> None:
    profile = resolve_model_runtime_profile(
        provider="ollama",
        model_name="gemma3:4b",
        role="reasoning",
        gpu_ram_gb=8.0,
    )

    assert profile.num_ctx == 8192
    assert profile.prompt_budget_tokens < profile.num_ctx
    assert profile.data_profile_budget_tokens > 0


def test_larger_local_model_uses_safer_context_on_8gb_gpu() -> None:
    profile = resolve_model_runtime_profile(
        provider="ollama",
        model_name="qwen3:8b",
        role="reasoning",
        gpu_ram_gb=8.0,
    )

    assert profile.num_ctx == 4096


def test_model_role_config_resolves_runtime_profile_from_settings() -> None:
    config = ModelRoleConfig(provider="ollama", model="gemma3:4b").with_runtime_profile(
        settings=ViRAGESettings(gpu_ram_gb=8.0),
        role="spec",
    )

    assert config.num_ctx == 8192
    assert config.max_output_tokens == 1024
    assert config.runtime_profile["role"] == "spec"


def test_prompt_budget_compresses_low_priority_sections() -> None:
    profile = resolve_model_runtime_profile(
        provider="ollama",
        model_name="gemma3:4b",
        role="reasoning",
        gpu_ram_gb=8.0,
        explicit_num_ctx=1024,
        explicit_prompt_budget_tokens=300,
    )
    prompt, report = build_budgeted_prompt(
        [
            PromptSection("required", "short required text", priority=0),
            PromptSection("long", "x" * 3000, min_tokens=64, priority=5),
        ],
        profile=profile,
    )

    assert estimate_tokens(prompt) <= 300
    assert report.was_compressed is True
