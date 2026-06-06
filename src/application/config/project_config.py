from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.llm.model_runtime import resolve_model_runtime_profile
from src.application.config.bootstrap import bootstrap_project_environment
from src.application.config.settings import ViRAGESettings


class ModelRoleConfig(BaseModel):
    provider: Literal["ollama", "openai", "huggingface"]
    model: str
    temperature: float = 0.0
    base_url: str | None = None
    timeout_seconds: float = 60.0
    num_ctx: int | None = None
    max_output_tokens: int | None = None
    prompt_budget_tokens: int | None = None
    runtime_profile: dict[str, object] = Field(default_factory=dict)

    def with_runtime_profile(self, *, settings: ViRAGESettings, role: str) -> "ModelRoleConfig":
        profile = resolve_model_runtime_profile(
            provider=self.provider,
            model_name=self.model,
            role=role,
            gpu_ram_gb=settings.gpu_ram_gb,
            explicit_num_ctx=self.num_ctx,
            explicit_max_output_tokens=self.max_output_tokens,
            explicit_prompt_budget_tokens=self.prompt_budget_tokens,
        )
        return self.model_copy(
            update={
                "num_ctx": profile.num_ctx,
                "max_output_tokens": profile.max_output_tokens,
                "prompt_budget_tokens": profile.prompt_budget_tokens,
                "runtime_profile": profile.as_dict(),
            }
        )


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


DEFAULT_CONFIG_PATH = Path("ui/config/app/project.toml")
EXAMPLE_CONFIG_PATH = Path("ui/config/app/project.example.toml")


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
    settings = ViRAGESettings(**payload.get("settings", {}))
    reasoning_model = ModelRoleConfig(**payload["reasoning_model"]).with_runtime_profile(settings=settings, role="reasoning")
    spec_model = ModelRoleConfig(**payload["spec_model"]).with_runtime_profile(settings=settings, role="spec")
    vlm_model = ModelRoleConfig(**payload["vlm_model"]).with_runtime_profile(settings=settings, role="vlm")
    return ProjectConfig(
        mode=payload.get("mode", "pipeline"),
        settings=settings,
        streamlit=StreamlitConfig(**payload.get("streamlit", {})),
        reasoning_model=reasoning_model,
        spec_model=spec_model,
        vlm_model=vlm_model,
    )
