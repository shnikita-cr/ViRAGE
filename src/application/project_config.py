from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.application.bootstrap import bootstrap_project_environment
from src.application.settings import ViRAGESettings


class ModelRoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "openai"]
    model: str
    temperature: float = 0.0
    base_url: str | None = None
    timeout_seconds: float = 60.0


class StreamlitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compute_metrics: bool = False
    show_step_logs: bool = True


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["pipeline", "streamlit", "benchmark"] = "pipeline"
    settings: ViRAGESettings = Field(default_factory=ViRAGESettings)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    reasoning_model: ModelRoleConfig
    spec_model: ModelRoleConfig
    vlm_model: ModelRoleConfig
    vision_judge_model: ModelRoleConfig


DEFAULT_CONFIG_PATH = Path("config/project.toml")
EXAMPLE_CONFIG_PATH = Path("config/project.example.toml")
_FORBIDDEN_SECRET_KEYS = {"api_key", "api_key_env", "token", "secret", "password"}


def _ensure_no_embedded_secrets(payload: Any, path: str = "root") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).strip().lower()
            if lowered in _FORBIDDEN_SECRET_KEYS or lowered.endswith("_key") or lowered.endswith("_token"):
                raise ValueError(
                    f"Secrets are forbidden in project config: found '{key}' at {path}. "
                    "Put credentials into .env or the process environment instead."
                )
            _ensure_no_embedded_secrets(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            _ensure_no_embedded_secrets(item, f"{path}[{index}]")


def load_project_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    bootstrap_project_environment()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Project configuration file was not found: {config_path}. "
            f"Copy {EXAMPLE_CONFIG_PATH} to {config_path} and edit it."
        )
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    _ensure_no_embedded_secrets(payload)
    return ProjectConfig(
        mode=payload.get("mode", "pipeline"),
        settings=ViRAGESettings(**payload.get("settings", {})),
        streamlit=StreamlitConfig(**payload.get("streamlit", {})),
        reasoning_model=ModelRoleConfig(**payload["reasoning_model"]),
        spec_model=ModelRoleConfig(**payload["spec_model"]),
        vlm_model=ModelRoleConfig(**payload["vlm_model"]),
        vision_judge_model=ModelRoleConfig(**payload["vision_judge_model"]),
    )
