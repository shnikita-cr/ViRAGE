from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.application.bootstrap import bootstrap_project_environment
from src.application.settings import ViRAGESettings


class ModelRoleConfig(BaseModel):
    provider: Literal["ollama", "openai", "huggingface"]
    model: str
    temperature: float = 0.0
    base_url: str | None = None
    timeout_seconds: float = 60.0


class StreamlitConfig(BaseModel):
    compute_metrics: bool = True
    show_step_logs: bool = True


class ProjectConfig(BaseModel):
    mode: Literal["pipeline", "streamlit", "benchmark"] = "pipeline"
    settings: ViRAGESettings = Field(default_factory=ViRAGESettings)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    reasoning_model: ModelRoleConfig
    spec_model: ModelRoleConfig
    vlm_model: ModelRoleConfig
    vision_judge_model: ModelRoleConfig


DEFAULT_CONFIG_PATH = Path("ui/config/project-gemma4.toml")
EXAMPLE_CONFIG_PATH = Path("ui/config/project.example.toml")


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
    return ProjectConfig(
        mode=payload.get("mode", "pipeline"),
        settings=ViRAGESettings(**payload.get("settings", {})),
        streamlit=StreamlitConfig(**payload.get("streamlit", {})),
        reasoning_model=ModelRoleConfig(**payload["reasoning_model"]),
        spec_model=ModelRoleConfig(**payload["spec_model"]),
        vlm_model=ModelRoleConfig(**payload["vlm_model"]),
        vision_judge_model=ModelRoleConfig(**payload["vision_judge_model"]),
    )
