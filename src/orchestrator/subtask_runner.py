from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.config.project_config import ProjectConfig
from src.orchestrator.contracts.models import AnalysisPlan, OrchestratorSubrunResult

_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


class OrchestratorSubtaskRunner:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def run(self, *, parent_run_id: str, plan: AnalysisPlan) -> list[OrchestratorSubrunResult]:
        pipeline = ViRAGEPipeline.from_project_config(self.config)
        results: list[OrchestratorSubrunResult] = []
        for index, subtask in enumerate(plan.subtasks, start=1):
            suffix = _safe_slug(subtask.id)
            subrun_id = f"{parent_run_id}/subruns/{index:02d}_{suffix}"
            request = PipelineRequest(
                query=subtask.query,
                data_path=plan.data_path,
                run_id=subrun_id,
                user_context={
                    "orchestrator_parent_run_id": parent_run_id,
                    "analysis_subtask": subtask.model_dump(),
                    "original_user_query": plan.user_query,
                },
            )
            error: str | None = None
            try:
                result = pipeline.invoke(request)
                run_id = result.run_id
            except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
                run_id = subrun_id
                error = f"{type(exc).__name__}: {exc}"
            run_dir = pipeline.runtime.ensure_run_dir(subrun_id)
            report_path = run_dir / "run_report.json"
            report_status = _read_run_report_status(report_path)
            results.append(
                OrchestratorSubrunResult(
                    subtask_id=subtask.id,
                    run_id=run_id,
                    status=report_status or ("error" if error else "completed"),
                    run_dir=run_dir.as_posix(),
                    run_report_path=report_path.as_posix(),
                    error=error,
                )
            )
        return results


def _read_run_report_status(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"artifact_read_error:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return "artifact_read_error:InvalidReportShape"
    status = str(payload.get("status") or "").strip().lower()
    return status if status else None


def _safe_slug(value: str) -> str:
    slug = _SAFE_RE.sub("_", value).strip("_")
    return slug or "subtask"
