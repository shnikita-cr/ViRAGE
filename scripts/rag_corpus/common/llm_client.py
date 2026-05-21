from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class LLMClientConfig:
    provider: str
    model: str
    base_url: str | None = None
    timeout_seconds: float = 120.0
    temperature: float = 0.0


class LLMClient:
    def __init__(self, config: LLMClientConfig) -> None:
        self.config = config

    def invoke(self, prompt: str) -> str:
        provider = self.config.provider.strip().lower()
        if provider == "ollama":
            return self._invoke_ollama(prompt)
        if provider == "openai":
            return self._invoke_openai(prompt)
        raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _invoke_ollama(self, prompt: str) -> str:
        base_url = (self.config.base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": self.config.temperature},
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected Ollama response: {data}")
        return content

    def _invoke_openai(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider='openai'.")
        base_url = (self.config.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected OpenAI response: {data}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected OpenAI content: {data}")
        return content


def parse_json_payload(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    for candidate in (stripped,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    array_match = _JSON_ARRAY_RE.search(stripped)
    if array_match:
        return json.loads(array_match.group(0))
    object_match = _JSON_OBJECT_RE.search(stripped)
    if object_match:
        return json.loads(object_match.group(0))
    raise ValueError("LLM output does not contain valid JSON payload.")
