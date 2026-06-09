from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from src.benchmark.external.errors import ExternalAdapterError

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma3:4b"


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    host: str = DEFAULT_OLLAMA_HOST
    model: str = DEFAULT_OLLAMA_MODEL

    @property
    def litellm_model(self) -> str:
        return f"ollama/{self.model}"

    def as_litellm_config(self) -> dict[str, Any]:
        return {
            "model": self.litellm_model,
            "environ_var_name": "OLLAMA_API_KEY",
            "api_key": "ollama",
            "api_base": self.host,
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            "provider": "ollama",
            "model": self.model,
            "api_base": self.host,
        }


def assert_ollama_model_available(settings: OllamaSettings, *, timeout_seconds: float = 10.0) -> None:
    tags_url = urljoin(_normalized_host(settings.host), "/api/tags")
    request = urllib.request.Request(tags_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise ExternalAdapterError(f"Ollama is not available at {settings.host}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExternalAdapterError(f"Ollama /api/tags response is not valid JSON: {exc}") from exc
    models = payload.get("models")
    if not isinstance(models, list):
        raise ExternalAdapterError("Ollama /api/tags response does not contain a models list.")
    names = {item.get("name") for item in models if isinstance(item, dict) and isinstance(item.get("name"), str)}
    if settings.model not in names:
        available = ", ".join(sorted(names)) if names else "no models returned"
        raise ExternalAdapterError(
            f"Ollama model '{settings.model}' is not installed at {settings.host}. Available models: {available}."
        )


def _normalized_host(host: str) -> str:
    return host.rstrip("/")
