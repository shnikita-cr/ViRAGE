from __future__ import annotations

import os

from src.application.bootstrap import bootstrap_project_environment
from src.application.project_config import ModelRoleConfig


def build_chat_model(config: ModelRoleConfig) -> object:
    bootstrap_project_environment()
    provider = config.provider.lower().strip()
    if provider == 'ollama':
        return _build_ollama_model(config)
    if provider == 'openai':
        return _build_openai_model(config)
    if provider == 'huggingface':
        return _build_huggingface_model(config)
    raise ValueError(f'Unsupported model provider: {config.provider}')


def _build_ollama_model(config: ModelRoleConfig) -> object:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError("langchain_ollama is required to use provider='ollama'.") from exc

    kwargs: dict[str, object] = {'model': config.model, 'temperature': config.temperature}
    if config.base_url:
        kwargs['base_url'] = config.base_url
    api_key = os.getenv(config.api_key_env) if config.api_key_env else None
    if api_key:
        kwargs['client_kwargs'] = {'headers': {'Authorization': f'Bearer {api_key}'}}
    return ChatOllama(**kwargs)


def _build_openai_model(config: ModelRoleConfig) -> object:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("langchain_openai is required to use provider='openai'.") from exc

    api_key_env = config.api_key_env or 'OPENAI_API_KEY'
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is required for provider='openai'.")

    kwargs: dict[str, object] = {
        'model': config.model,
        'temperature': config.temperature,
        'api_key': api_key,
        'timeout': config.timeout_seconds,
    }
    if config.base_url:
        kwargs['base_url'] = config.base_url
    return ChatOpenAI(**kwargs)


def _build_huggingface_model(config: ModelRoleConfig) -> object:
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    except ImportError as exc:
        raise RuntimeError("langchain_huggingface is required to use provider='huggingface'.") from exc

    api_key = os.getenv(config.api_key_env or 'HUGGINGFACEHUB_API_TOKEN')
    endpoint_kwargs: dict[str, object] = {
        'repo_id': config.model,
        'temperature': config.temperature,
        'timeout': config.timeout_seconds,
    }
    if api_key:
        endpoint_kwargs['huggingfacehub_api_token'] = api_key
    if config.base_url:
        endpoint_kwargs['endpoint_url'] = config.base_url
    return ChatHuggingFace(llm=HuggingFaceEndpoint(**endpoint_kwargs))
