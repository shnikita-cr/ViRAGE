from __future__ import annotations

from src.llm.model_runtime import build_model_runtime_profile


def test_gemma3_4b_uses_extended_context_on_8gb() -> None:
    profile = build_model_runtime_profile(
        provider="ollama",
        model_name="gemma3:4b",
        role="spec",
        gpu_ram_gb=8.0,
    )

    assert profile.num_ctx == 8192
    assert profile.max_output_tokens >= 768
    assert profile.prompt_budget_tokens < profile.num_ctx


def test_7b_model_uses_safe_context_on_8gb() -> None:
    profile = build_model_runtime_profile(
        provider="ollama",
        model_name="qwen2.5-coder:7b",
        role="spec",
        gpu_ram_gb=8.0,
    )

    assert profile.num_ctx == 4096


def test_configured_num_ctx_overrides_dynamic_heuristic() -> None:
    profile = build_model_runtime_profile(
        provider="ollama",
        model_name="gemma3:4b",
        role="spec",
        gpu_ram_gb=8.0,
        configured_num_ctx=4096,
        configured_max_output_tokens=512,
    )

    assert profile.num_ctx == 4096
    assert profile.max_output_tokens == 512
