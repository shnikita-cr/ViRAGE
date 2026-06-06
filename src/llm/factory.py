from __future__ import annotations

from src.application.config.bootstrap import bootstrap_project_environment
from src.application.config.project_config import ModelRoleConfig
from src.llm.model_runtime import attach_runtime_profile, resolve_model_runtime_profile


def build_chat_model(config: ModelRoleConfig) -> object:
    bootstrap_project_environment()
    provider = config.provider.lower().strip()
    if provider == "ollama":
        return _build_ollama_model(config)
    if provider == "openai":
        return _build_openai_model(config)
    if provider == "huggingface":
        return _build_huggingface_model(config)
    raise ValueError(f"Unsupported model provider: {config.provider}")


def _build_ollama_model(config: ModelRoleConfig) -> object:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError("langchain_ollama is required to use provider='ollama'.") from exc

    profile = _runtime_profile(config, role="reasoning")
    kwargs: dict[str, object] = {
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout_seconds,
        "num_ctx": profile.num_ctx,
        "num_predict": profile.max_output_tokens,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return attach_runtime_profile(ChatOllama(**kwargs), profile)


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
    if config.base_url:
        kwargs["base_url"] = config.base_url
    model = ChatOpenAI(**kwargs)
    return attach_runtime_profile(model, _runtime_profile(config, role="reasoning"))


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
    model = ChatHuggingFace(llm=HuggingFaceEndpoint(**endpoint_kwargs))
    return attach_runtime_profile(model, _runtime_profile(config, role="reasoning"))


def _runtime_profile(config: ModelRoleConfig, *, role: str):
    if config.runtime_profile:
        return resolve_model_runtime_profile(
            provider=config.provider,
            model_name=config.model,
            role=str(config.runtime_profile.get("role") or role),
            gpu_ram_gb=float(config.runtime_profile.get("gpu_ram_gb") or 8.0),
            explicit_num_ctx=int(config.runtime_profile.get("num_ctx") or config.num_ctx or 0) or None,
            explicit_max_output_tokens=int(config.runtime_profile.get("max_output_tokens") or config.max_output_tokens or 0) or None,
            explicit_prompt_budget_tokens=int(config.runtime_profile.get("prompt_budget_tokens") or config.prompt_budget_tokens or 0) or None,
        )
    return resolve_model_runtime_profile(
        provider=config.provider,
        model_name=config.model,
        role=role,
        explicit_num_ctx=config.num_ctx,
        explicit_max_output_tokens=config.max_output_tokens,
        explicit_prompt_budget_tokens=config.prompt_budget_tokens,
    )
