from __future__ import annotations

from typing import Any

from src.application.config.bootstrap import bootstrap_project_environment
from src.application.config.project_config import ModelRoleConfig


def build_chat_model(
    config: ModelRoleConfig,
    settings: object | None = None,
    *,
    role: str | None = None,
) -> object:
    bootstrap_project_environment()
    provider = config.provider.lower().strip()
    if provider == "ollama":
        return _build_ollama_model(config, settings=settings, role=role)
    if provider == "openai":
        return _build_openai_model(config)
    if provider == "huggingface":
        return _build_huggingface_model(config)
    raise ValueError(f"Unsupported model provider: {config.provider}")


def _build_ollama_model(config: ModelRoleConfig, *, settings: object | None, role: str | None) -> object:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError("langchain_ollama is required to use provider='ollama'.") from exc

    kwargs: dict[str, object] = {
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout_seconds,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    num_ctx = config.num_ctx or _default_num_ctx(config.model, settings=settings, role=role)
    max_output_tokens = config.max_output_tokens or _default_max_output_tokens(role)
    kwargs["num_ctx"] = int(num_ctx)
    kwargs["num_predict"] = int(max_output_tokens)
    return ChatOllama(**kwargs)


def _build_openai_model(config: ModelRoleConfig) -> object:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("langchain_openai is required to use provider='openai'.") from exc

    kwargs: dict[str, object] = {
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout_seconds,
    }
    if config.max_output_tokens:
        kwargs["max_tokens"] = int(config.max_output_tokens)
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenAI(**kwargs)


def _build_huggingface_model(config: ModelRoleConfig) -> object:
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    except ImportError as exc:
        raise RuntimeError("langchain_huggingface is required to use provider='huggingface'.") from exc

    endpoint_kwargs: dict[str, object] = {
        "repo_id": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout_seconds,
    }
    if config.max_output_tokens:
        endpoint_kwargs["max_new_tokens"] = int(config.max_output_tokens)
    if config.base_url:
        endpoint_kwargs["endpoint_url"] = config.base_url
    return ChatHuggingFace(llm=HuggingFaceEndpoint(**endpoint_kwargs))


def _default_num_ctx(model_name: str, *, settings: object | None, role: str | None) -> int:
    gpu_ram_gb = float(getattr(settings, "gpu_ram_gb", 8.0) or 8.0)
    lowered = model_name.lower()
    if gpu_ram_gb < 8.0:
        return 4096
    if any(token in lowered for token in ("12b", "14b", "70b")):
        return 4096
    if any(token in lowered for token in ("7b", "8b")):
        return 4096
    if "4b" in lowered or "e2b" in lowered or "e4b" in lowered:
        return 8192 if role in {"reasoning", "spec"} else 4096
    return 4096


def _default_max_output_tokens(role: str | None) -> int:
    if role == "spec":
        return 1000
    return 1000
