from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.application.settings import ViRAGESettings
from src.domain.models import ModelCallLog, TokenUsage


@dataclass(slots=True)
class RuntimeContext:
    settings: ViRAGESettings
    reasoning_llm: Any | None = None
    spec_llm: Any | None = None
    vlm: Any | None = None
    vision_judge_llm: Any | None = None
    codegen_llm: Any | None = None  # legacy compatibility only
    model_call_logs: list[ModelCallLog] = field(default_factory=list)

    def ensure_run_dir(self, run_id: str) -> Path:
        path = self.settings.artifact_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def reset_model_logs(self) -> None:
        self.model_call_logs.clear()

    def add_model_call_log(self, log: ModelCallLog) -> None:
        self.model_call_logs.append(log)

    def token_usage_summary(self) -> TokenUsage:
        summary = TokenUsage()
        for item in self.model_call_logs:
            summary.prompt_tokens += item.token_usage.prompt_tokens
            summary.completion_tokens += item.token_usage.completion_tokens
            summary.total_tokens += item.token_usage.total_tokens
        return summary
