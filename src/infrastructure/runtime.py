from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.application.settings import ViRAGESettings
from src.application.dataset_context import read_dataframe_cached
from src.domain.models import ModelCallLog, StageExecutionLog, StepLog, TokenUsage

_MODEL_CALL_CSV_COLUMNS = [
    "call_index",
    "stage",
    "model_role",
    "provider",
    "model_name",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "duration_ms",
    "duration_seconds",
    "started_at",
    "finished_at",
    "attempts",
    "attempt_number",
    "parser_error_count",
    "has_error",
]

_STAGE_CSV_COLUMNS = [
    "execution_index",
    "node_name",
    "pipeline_stage",
    "status",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "duration_ms",
    "duration_seconds",
    "started_at",
    "finished_at",
    "model_call_start_index",
    "model_call_end_index",
    "model_call_count",
    "error",
]

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(slots=True)
class RuntimeContext:
    settings: ViRAGESettings
    reasoning_llm: Any | None = None
    spec_llm: Any | None = None
    vlm: Any | None = None
    vision_judge_llm: Any | None = None
    model_call_logs: list[ModelCallLog] = field(default_factory=list)
    stage_execution_logs: list[StageExecutionLog] = field(default_factory=list)
    artifact_indices: dict[str, int] = field(default_factory=dict)
    current_run_id: str | None = None
    step_callback: Callable[[StepLog], None] | None = None
    model_call_callback: Callable[[ModelCallLog], None] | None = None
    dataframe_cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def ensure_run_dir(self, run_id: str | None = None) -> Path:
        rid = run_id or self.current_run_id
        if not rid:
            raise RuntimeError("RuntimeContext.current_run_id is not set.")
        path = self.settings.artifact_root / rid
        path.mkdir(parents=True, exist_ok=True)
        return path

    def read_dataframe(self, path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
        return read_dataframe_cached(self.dataframe_cache, path, nrows=nrows)

    @staticmethod
    def _remove_paths(paths: list[Path]) -> None:
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()

    def reset_model_logs(self) -> None:
        self.model_call_logs.clear()
        try:
            run_dir = self.ensure_run_dir()
        except Exception:
            return
        self._remove_paths([
            run_dir / "model_calls",
            run_dir / "model_calls.csv",
            run_dir / "model_call_tokens.csv",
            run_dir / "model_call_timings.csv",
            run_dir / "artifacts" / "model_call_logs.json",
        ])

    def reset_stage_execution_logs(self) -> None:
        self.stage_execution_logs.clear()
        try:
            run_dir = self.ensure_run_dir()
        except Exception:
            return
        self._remove_paths([
            run_dir / "stages.csv",
            run_dir / "stage_timings.csv",
            run_dir / "stage_tokens.csv",
            run_dir / "stage_executions",
            run_dir / "artifacts",
            run_dir / "nodes",
        ])

    def reset_artifact_indices(self, *, run_id: str | None = None) -> None:
        rid = run_id or self.current_run_id
        if rid:
            self.artifact_indices[rid] = 0

    def next_artifact_path(self, relative_path: str, *, run_id: str | None = None) -> Path:
        return self.ensure_run_dir(run_id) / self._numbered_relative_path(relative_path, run_id=run_id)

    def add_model_call_log(self, log: ModelCallLog) -> None:
        index = len(self.model_call_logs) + 1
        enriched_log = log.model_copy(update={"call_index": index})
        self.model_call_logs.append(enriched_log)
        try:
            run_dir = self.ensure_run_dir()
        except Exception:
            run_dir = None
        if run_dir is not None:
            calls_dir = run_dir / "model_calls"
            calls_dir.mkdir(parents=True, exist_ok=True)
            path = calls_dir / self._model_call_filename(enriched_log)
            path.write_text(json.dumps(enriched_log.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_model_call_csvs(run_dir)
        if self.model_call_callback:
            self.model_call_callback(enriched_log)

    def add_stage_execution_log(self, log: StageExecutionLog) -> StageExecutionLog:
        index = len(self.stage_execution_logs) + 1
        enriched_log = log.model_copy(update={"execution_index": index})
        self.stage_execution_logs.append(enriched_log)
        try:
            run_dir = self.ensure_run_dir()
        except Exception:
            run_dir = None
        if run_dir is not None:
            self._write_stage_execution_csvs(run_dir)
        return enriched_log

    def token_usage_summary(self) -> TokenUsage:
        summary = TokenUsage()
        for item in self.model_call_logs:
            summary.prompt_tokens += item.token_usage.prompt_tokens
            summary.completion_tokens += item.token_usage.completion_tokens
            summary.total_tokens += item.token_usage.total_tokens
        return summary

    def save_model_log_artifacts(self, *, run_id: str | None = None) -> None:
        run_dir = self.ensure_run_dir(run_id)
        self._write_model_call_csvs(run_dir)
        self._write_stage_execution_csvs(run_dir)
        self._remove_paths([run_dir / "artifacts" / "model_call_logs.json"])

    def save_run_status(
            self,
            *,
            run_id: str | None = None,
            status: str,
            final_stage: str | None = None,
            semantic_status: str | None = None,
            error_type: str | None = None,
            error: str | None = None,
            extra: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "status": status,
            "final_stage": final_stage,
            "semantic_status": semantic_status,
            "error_type": error_type,
            "error": error,
            "has_model_calls_csv": (self.ensure_run_dir(run_id) / "model_calls.csv").exists(),
            "has_stages_csv": (self.ensure_run_dir(run_id) / "stages.csv").exists(),
            **(extra or {}),
        }
        return self.save_json_artifact("run_status.json", payload, run_id=run_id)

    def save_json_artifact(
            self,
            relative_path: str,
            payload: Any,
            *,
            run_id: str | None = None,
            numbered: bool = False,
    ) -> str:
        path = self.next_artifact_path(relative_path, run_id=run_id) if numbered else self.ensure_run_dir(
            run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path.as_posix()

    def save_text_artifact(
            self,
            relative_path: str,
            text: str,
            *,
            run_id: str | None = None,
            numbered: bool = False,
    ) -> str:
        path = self.next_artifact_path(relative_path, run_id=run_id) if numbered else self.ensure_run_dir(
            run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path.as_posix()

    def save_bytes_artifact(
            self,
            relative_path: str,
            data: bytes,
            *,
            run_id: str | None = None,
            numbered: bool = False,
    ) -> str:
        path = self.next_artifact_path(relative_path, run_id=run_id) if numbered else self.ensure_run_dir(
            run_id) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path.as_posix()

    def emit_step(self, log: StepLog) -> None:
        if self.step_callback:
            self.step_callback(log)

    def _numbered_relative_path(self, relative_path: str, *, run_id: str | None = None) -> Path:
        rid = run_id or self.current_run_id
        if not rid:
            raise RuntimeError("RuntimeContext.current_run_id is not set.")
        index = self.artifact_indices.get(rid, 0) + 1
        self.artifact_indices[rid] = index

        path = Path(relative_path)
        filename = path.name
        prefix = f"{index:03d}_"
        if filename.startswith(prefix):
            numbered_name = filename
        else:
            numbered_name = f"{prefix}{filename}"
        return path.with_name(numbered_name)

    def _write_model_call_csvs(self, run_dir: Path) -> None:
        """Write one compact CSV with model-call token and timing statistics.

        Detailed prompts/responses remain in model_calls/*.json. The CSV is meant
        for analysis and benchmarking, so it contains only scalar fields.
        """
        path = run_dir / "model_calls.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=_MODEL_CALL_CSV_COLUMNS)
            writer.writeheader()
            for log in self.model_call_logs:
                writer.writerow(self._model_call_csv_row(log))

        self._remove_paths([run_dir / "model_call_tokens.csv", run_dir / "model_call_timings.csv"])

    def _write_stage_execution_csvs(self, run_dir: Path) -> None:
        """Write one compact CSV with stage token and timing statistics."""
        path = run_dir / "stages.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=_STAGE_CSV_COLUMNS)
            writer.writeheader()
            for log in self.stage_execution_logs:
                writer.writerow(self._stage_csv_row(log))

        self._remove_paths([run_dir / "stage_timings.csv", run_dir / "stage_tokens.csv"])

    @staticmethod
    def _model_call_filename(log: ModelCallLog) -> str:
        call_index = log.call_index or 0
        stage = RuntimeContext._safe_filename_part(log.stage)
        role = RuntimeContext._safe_filename_part(log.model_role)
        attempt_number = max(1, log.attempt_number)
        return f"{call_index:03d}_{stage}_{role}-{attempt_number:02d}.json"

    @staticmethod
    def _safe_filename_part(value: str) -> str:
        cleaned = _FILENAME_SAFE_RE.sub("_", value.strip()).strip("_")
        return cleaned or "unknown"

    @staticmethod
    def _model_call_csv_row(log: ModelCallLog) -> dict[str, Any]:
        return {
            "call_index": log.call_index,
            "stage": log.stage,
            "model_role": log.model_role,
            "provider": log.provider or "",
            "model_name": log.model_name,
            "prompt_tokens": log.token_usage.prompt_tokens,
            "completion_tokens": log.token_usage.completion_tokens,
            "total_tokens": log.token_usage.total_tokens,
            "duration_ms": log.duration_ms,
            "duration_seconds": log.duration_seconds,
            "started_at": log.started_at or "",
            "finished_at": log.finished_at or "",
            "attempts": log.attempts,
            "attempt_number": log.attempt_number,
            "parser_error_count": len(log.parser_errors),
            "has_error": bool(log.parser_errors),
        }

    @staticmethod
    def _stage_csv_row(log: StageExecutionLog) -> dict[str, Any]:
        return {
            "execution_index": log.execution_index,
            "node_name": log.node_name,
            "pipeline_stage": log.pipeline_stage or "",
            "status": log.status,
            "prompt_tokens": log.token_usage.prompt_tokens,
            "completion_tokens": log.token_usage.completion_tokens,
            "total_tokens": log.token_usage.total_tokens,
            "duration_ms": log.duration_ms,
            "duration_seconds": log.duration_seconds,
            "started_at": log.started_at,
            "finished_at": log.finished_at,
            "model_call_start_index": log.model_call_start_index,
            "model_call_end_index": log.model_call_end_index,
            "model_call_count": log.model_call_count,
            "error": (log.error or "")[:500],
        }
