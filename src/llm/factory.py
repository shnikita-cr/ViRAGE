from __future__ import annotations

from src.application.config.bootstrap import bootstrap_project_environment
from src.application.config.project_config import ModelRoleConfig
from src.application.config.settings import ViRAGESettings
from src.llm.model_runtime import attach_runtime_profile, build_model_runtime_profile


def build_chat_model(config: ModelRoleConfig, settings: ViRAGESettings | None = None, *, role: str = "reasoning") -> object:
    bootstrap_project_environment()
    provider = config.provider.lower().strip()
    if provider == "ollama":
        return _build_ollama_model(config, settings, role=role)
    if provider == "openai":
        return _build_openai_model(config, settings, role=role)
    if provider == "huggingface":
        return _build_huggingface_model(config)
    raise ValueError(f"Unsupported model provider: {config.provider}")


def _build_ollama_model(config: ModelRoleConfig, settings: ViRAGESettings | None, *, role: str) -> object:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError("langchain_ollama is required to use provider='ollama'.") from exc

    runtime_profile = build_model_runtime_profile(
        provider=config.provider,
        model_name=config.model,
        role=role,
        gpu_ram_gb=float(settings.gpu_ram_gb if settings is not None else 8.0),
        configured_num_ctx=config.num_ctx,
        configured_max_output_tokens=config.max_output_tokens,
    )
    kwargs: dict[str, object] = {
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout_seconds,
        "num_ctx": runtime_profile.num_ctx,
        "num_predict": runtime_profile.max_output_tokens,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return attach_runtime_profile(ChatOllama(**kwargs), runtime_profile)


def _build_openai_model(config: ModelRoleConfig, settings: ViRAGESettings | None, *, role: str) -> object:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("langchain_openai is required to use provider='openai'.") from exc

    runtime_profile = build_model_runtime_profile(
        provider=config.provider,
        model_name=config.model,
        role=role,
        gpu_ram_gb=float(settings.gpu_ram_gb if settings is not None else 8.0),
        configured_num_ctx=config.num_ctx,
        configured_max_output_tokens=config.max_output_tokens,
    )
    kwargs: dict[str, object] = {
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout_seconds,
    }
    if config.max_output_tokens is not None:
        kwargs["max_tokens"] = config.max_output_tokens
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return attach_runtime_profile(ChatOpenAI(**kwargs), runtime_profile)


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
    if config.base_url:
        endpoint_kwargs["endpoint_url"] = config.base_url
    return ChatHuggingFace(llm=HuggingFaceEndpoint(**endpoint_kwargs))
