from __future__ import annotations

import os
from typing import Any

from src.application.bootstrap import bootstrap_project_environment
from src.application.project_config import ModelRoleConfig


def build_chat_model(config: ModelRoleConfig) -> object:
    bootstrap_project_environment()
    provider = config.provider.lower().strip()
    if provider == "ollama":
        return _build_ollama_model(config)
    if provider == "openai":
        return _build_openai_model(config)
    raise ValueError(f"Unsupported model provider: {config.provider}")


def _ollama_auth_headers() -> dict[str, str]:
    api_key = os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_CLOUD_API_KEY")
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _build_ollama_model(config: ModelRoleConfig) -> object:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError("langchain_ollama is required to use provider='ollama'.") from exc

    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url

    headers = _ollama_auth_headers()
    if headers:
        # ChatOllama forwards client_kwargs to the underlying httpx client in modern releases.
        kwargs["client_kwargs"] = {"headers": headers}

    return ChatOllama(**kwargs)


def _build_openai_model(config: ModelRoleConfig) -> object:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("langchain_openai is required to use provider='openai'.") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable OPENAI_API_KEY is required for provider='openai'.")

    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "api_key": api_key,
        "timeout": config.timeout_seconds,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenAI(**kwargs)
