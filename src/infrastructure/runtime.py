from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.application.settings import ViRAGESettings
from src.domain.models import ModelCallLog, StepLog, TokenUsage


@dataclass(slots=True)
class RuntimeContext:
    settings: ViRAGESettings
    reasoning_llm: Any | None = None
    spec_llm: Any | None = None
    vlm: Any | None = None
    vision_judge_llm: Any | None = None
    model_call_logs: list[ModelCallLog] = field(default_factory=list)
    current_run_id: str | None = None
    step_callback: Callable[[StepLog], None] | None = None
    model_call_callback: Callable[[ModelCallLog], None] | None = None

    def ensure_run_dir(self, run_id: str | None = None) -> Path:
        rid = run_id or self.current_run_id
        if not rid:
            raise RuntimeError('RuntimeContext.current_run_id is not set.')
        path = self.settings.artifact_root / rid
        path.mkdir(parents=True, exist_ok=True)
        return path

    def reset_model_logs(self) -> None:
        self.model_call_logs.clear()

    def add_model_call_log(self, log: ModelCallLog) -> None:
        self.model_call_logs.append(log)
        try:
            run_dir = self.ensure_run_dir()
        except Exception:
            run_dir = None
        if run_dir is not None:
            calls_dir = run_dir / 'model_calls'
            calls_dir.mkdir(parents=True, exist_ok=True)
            index = len(self.model_call_logs)
            path = calls_dir / f'{index:03d}_{log.stage}_{log.model_role}.json'
            path.write_text(json.dumps(log.model_dump(), ensure_ascii=False, indent=2), encoding='utf-8')
        if self.model_call_callback:
            self.model_call_callback(log)

    def token_usage_summary(self) -> TokenUsage:
        summary = TokenUsage()
        for item in self.model_call_logs:
            summary.prompt_tokens += item.token_usage.prompt_tokens
            summary.completion_tokens += item.token_usage.completion_tokens
            summary.total_tokens += item.token_usage.total_tokens
        return summary

    def save_json_artifact(self, relative_path: str, payload: Any, *, run_id: str | None = None) -> str:
        path = self.ensure_run_dir(run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        return path.as_posix()

    def save_text_artifact(self, relative_path: str, text: str, *, run_id: str | None = None) -> str:
        path = self.ensure_run_dir(run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path.as_posix()

    def save_bytes_artifact(self, relative_path: str, data: bytes, *, run_id: str | None = None) -> str:
        path = self.ensure_run_dir(run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path.as_posix()

    def emit_step(self, log: StepLog) -> None:
        if self.step_callback:
            self.step_callback(log)
