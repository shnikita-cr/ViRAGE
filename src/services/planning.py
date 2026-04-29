from __future__ import annotations

from src.domain.models import (
    AnalysisRubric,
    DataProfile,
    ExecutionPolicy,
    PlanningResult,
    PlanningStep,
    QueryUnderstandingResult,
    RequestAnalysisResult,
    ValidationPolicy,
    VisRAGResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class PlanningService(BaseService):
    """Deterministic policy builder. No LLM call."""

    def invoke(
        self,
        query_understanding: QueryUnderstandingResult,
        request_analysis: RequestAnalysisResult,
        data_profile: DataProfile,
        visrag: VisRAGResult,
        runtime: RuntimeContext,
    ) -> PlanningResult:
        return _build_plan(query_understanding, request_analysis, data_profile, visrag)

    async def ainvoke(
        self,
        query_understanding: QueryUnderstandingResult,
        request_analysis: RequestAnalysisResult,
        data_profile: DataProfile,
        visrag: VisRAGResult,
        runtime: RuntimeContext,
    ) -> PlanningResult:
        return self.invoke(query_understanding, request_analysis, data_profile, visrag, runtime)


def _build_plan(
    query_understanding: QueryUnderstandingResult,
    request_analysis: RequestAnalysisResult,
    data_profile: DataProfile,
    visrag: VisRAGResult,
) -> PlanningResult:
    return PlanningResult(
        steps=[
            PlanningStep(name="use_visrag_candidate", description="Use the selected prepared-corpus VisRAG candidate."),
            PlanningStep(name="validate_spec", description="Validate Vega-Lite before rendering."),
            PlanningStep(name="analyze_rendered_chart", description="Analyze only the validated rendered chart."),
        ],
        success_criteria=[
            "Vega-Lite specification is valid.",
            "Rendered chart is not empty.",
            "Insights are supported by visible chart evidence or data calculations.",
        ],
        execution_policy=ExecutionPolicy(max_retries=0, retry_strategy="fail_fast", prefer_best_ranked_spec=True),
        validation_policy=ValidationPolicy(use_spec_validator=True, use_scenegraph_check=True, use_empty_chart_check=True, fail_fast_on_schema_error=True),
        analysis_rubric=AnalysisRubric(
            focus_areas=_focus_areas(query_understanding, request_analysis, data_profile, visrag),
            output_format="bullet_points",
            strict_visual_only=True,
            emphasize_anomalies=True,
        ),
    )


def _focus_areas(
    query_understanding: QueryUnderstandingResult,
    request_analysis: RequestAnalysisResult,
    data_profile: DataProfile,
    visrag: VisRAGResult,
) -> list[str]:
    values = [query_understanding.task_type or "", query_understanding.analysis_goal or "", *request_analysis.selected_fields]
    if data_profile.likely_time_columns:
        values.append("trend")
    if data_profile.likely_numeric_columns:
        values.append("numeric_values")
    return list(dict.fromkeys(item for item in values if item))
