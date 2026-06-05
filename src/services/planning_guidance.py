from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.domain.models import DataProfile, QueryRequestAnalysisResult, VisRAGResult
from src.infrastructure.runtime import RuntimeContext
from src.services.visrag import VisRAGService


@dataclass(frozen=True)
class PlanningGuidanceResult:
    text: str
    visrag: VisRAGResult | None = None


class PlanningGuidanceService:
    def invoke(
            self,
            *,
            user_query: str,
            data_profile: DataProfile,
            runtime: RuntimeContext | None,
            input_type: str,
    ) -> PlanningGuidanceResult:
        if runtime is None or not bool(getattr(runtime.settings, "visrag_enabled", False)):
            return PlanningGuidanceResult(text="")
        if not self._runtime_corpus_available(runtime):
            return PlanningGuidanceResult(text="")
        available_fields = [column.name for column in data_profile.columns]
        query_analysis = QueryRequestAnalysisResult(
            normalized_query=user_query.strip(),
            analysis_task="analysis_planning",
            selected_fields=available_fields[:30],
            query_variants=[],
            confidence=0.8,
        )
        result = VisRAGService().invoke(
            query_analysis,
            data_profile,
            runtime=runtime,
            task_context={
                "stage": "planning",
                "input_type": input_type,
                "purpose": "choose analytical subtasks, metric semantics, ranking strategy and visual constraints",
                "available_fields": available_fields[:40],
                "preferred_source_kinds": [
                    "planner_guidance",
                    "metric_semantics_guidance",
                    "image_folder_guidance",
                    "multi_metric_guidance",
                    "quality_severity_guidance",
                    "eda_guidance",
                ],
            },
        )
        if getattr(runtime, "current_run_id", None):
            runtime.save_json_artifact(
                "nodes/planning_guidance.json",
                result.model_dump(),
                run_id=runtime.current_run_id,
                numbered=True,
            )
        return PlanningGuidanceResult(text=result.generation_guidance.prompt_text.strip(), visrag=result)
    @staticmethod
    def _runtime_corpus_available(runtime: RuntimeContext) -> bool:
        corpus_root = getattr(runtime.settings, "visrag_corpus_root", None)
        if corpus_root is None:
            return False
        path = Path(corpus_root)
        if path.is_file():
            return True
        return (path / "guidance_chunks.jsonl").is_file()

