from __future__ import annotations

import re

from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.project_config import ProjectConfig
from src.orchestrator.models import AnalysisPlan, OrchestratorSubrunResult

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
            result = pipeline.invoke(request)
            run_dir = pipeline.runtime.ensure_run_dir(subrun_id)
            results.append(
                OrchestratorSubrunResult(
                    subtask_id=subtask.id,
                    run_id=result.run_id,
                    status="completed",
                    run_dir=run_dir.as_posix(),
                    run_report_path=(run_dir / "run_report.json").as_posix(),
                )
            )
        return results


def _safe_slug(value: str) -> str:
    slug = _SAFE_RE.sub("_", value).strip("_")
    return slug or "subtask"
