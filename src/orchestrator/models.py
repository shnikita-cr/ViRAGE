from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, model_validator

from src.orchestrator.planning_contract import (
    MetricSemantic,
    RankingStrategy,
    ScaleStrategy,
    VisualConstraint,
    allowed_metric_semantics,
    allowed_ranking_strategies,
    allowed_scale_strategies,
    allowed_visual_constraints,
    is_problem_ranking_strategy,
    requires_problem_ranking,
    requires_severity_fields,
    supports_severity_scale,
)

AnalysisTaskType = Literal[
    "overview",
    "distribution",
    "group_comparison",
    "correlation",
    "temporal_trend",
    "outlier_detection",
    "ranking",
    "missingness_analysis",
    "image_quality_analysis",
]

_ALLOWED_TASK_TYPES = set(get_args(AnalysisTaskType))


class AnalysisSubtask(BaseModel):
    id: str = Field(min_length=1)
    task_type: AnalysisTaskType
    query: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    required_fields: list[str] = Field(min_length=1)
    optional_fields: list[str] = Field(default_factory=list)
    priority: int = Field(default=100, ge=1)
    constraints: dict[str, Any] = Field(default_factory=dict)
    metric_semantics: dict[str, MetricSemantic] = Field(default_factory=dict)
    ranking_strategy: RankingStrategy | None = None
    scale_strategy: ScaleStrategy | None = None
    visual_constraints: list[VisualConstraint] = Field(default_factory=list)
    comparison_group_id: str | None = None
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _dedupe_fields(self) -> "AnalysisSubtask":
        self.required_fields = _dedupe(self.required_fields)
        self.optional_fields = [field for field in _dedupe(self.optional_fields) if field not in self.required_fields]
        self.visual_constraints = _dedupe(self.visual_constraints)
        self.metric_semantics = {str(key).strip(): value for key, value in dict(self.metric_semantics or {}).items() if str(key).strip()}
        if not self.required_fields:
            raise ValueError(f"Analysis subtask {self.id!r} must contain at least one required field.")
        self._validate_problematic_strategy()
        return self

    def _validate_problematic_strategy(self) -> None:
        if not requires_problem_ranking(self.query, self.purpose, self.task_type):
            return
        if not is_problem_ranking_strategy(self.ranking_strategy):
            raise ValueError(
                f"Analysis subtask {self.id!r} targets problematic/quality items but has no controlled ranking_strategy. "
                f"Allowed ranking strategies: {allowed_ranking_strategies()}"
            )
        if requires_severity_fields(self.ranking_strategy) and not self.metric_semantics:
            raise ValueError(
                f"Analysis subtask {self.id!r} uses severity ranking but metric_semantics is empty. "
                f"Allowed metric semantics: {allowed_metric_semantics()}"
            )
        if requires_severity_fields(self.ranking_strategy) and not supports_severity_scale(self.scale_strategy):
            raise ValueError(
                f"Analysis subtask {self.id!r} uses severity ranking but scale_strategy={self.scale_strategy!r}. "
                f"Allowed severity scale strategies: {sorted({'normalized_severity', 'independent_panels', 'single_metric'})}"
            )


class SkippedAnalysisCandidate(BaseModel):
    task_type: AnalysisTaskType
    reason: str = Field(min_length=1)
    required_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _dedupe_fields(self) -> "SkippedAnalysisCandidate":
        self.required_fields = _dedupe(self.required_fields)
        return self


class AnalysisPlan(BaseModel):
    user_query: str = Field(min_length=1)
    data_path: str = Field(min_length=1)
    input_type: str = "table"
    original_input_path: str | None = None
    preprocessing_report_path: str | None = None
    max_charts: int = Field(default=3, ge=1, le=3)
    subtasks: list[AnalysisSubtask] = Field(min_length=1, max_length=3)
    skipped_candidates: list[SkippedAnalysisCandidate]
    rationale: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_subtasks(self) -> "AnalysisPlan":
        if len(self.subtasks) > self.max_charts:
            raise ValueError(f"AnalysisPlan has {len(self.subtasks)} subtasks, max_charts={self.max_charts}.")
        ids = [item.id for item in self.subtasks]
        duplicates = {item for item in ids if ids.count(item) > 1}
        if duplicates:
            raise ValueError(f"Duplicate analysis subtask ids: {sorted(duplicates)}")
        return self

    def validate_against_available_fields(self, available_fields: set[str]) -> "AnalysisPlan":
        if not available_fields:
            raise ValueError("AnalysisPlan validation requires at least one available dataset field.")
        for subtask in self.subtasks:
            _validate_fields(
                fields=subtask.required_fields,
                available_fields=available_fields,
                context=f"subtask {subtask.id!r} required_fields",
            )
            _validate_fields(
                fields=subtask.optional_fields,
                available_fields=available_fields,
                context=f"subtask {subtask.id!r} optional_fields",
            )
            _validate_fields(
                fields=list(subtask.metric_semantics.keys()),
                available_fields=available_fields,
                context=f"subtask {subtask.id!r} metric_semantics",
            )
        for candidate in self.skipped_candidates:
            _validate_fields(
                fields=candidate.required_fields,
                available_fields=available_fields,
                context=f"skipped candidate {candidate.task_type!r} required_fields",
            )
        return self


class OrchestratorSubrunResult(BaseModel):
    subtask_id: str
    run_id: str
    status: str
    run_dir: str
    run_report_path: str | None = None
    error: str | None = None


class OrchestratorReport(BaseModel):
    run_id: str
    user_query: str
    data_path: str
    input_type: str = "table"
    original_input_path: str | None = None
    preprocessing_report_path: str | None = None
    plan_path: str
    data_profile_path: str | None = None
    executed: bool = False
    subruns: list[OrchestratorSubrunResult] = Field(default_factory=list)
    final_summary_path: str | None = None


def allowed_analysis_task_types() -> list[str]:
    return sorted(_ALLOWED_TASK_TYPES)


def analysis_plan_contract_metadata() -> dict[str, list[str]]:
    return {
        "task_types": allowed_analysis_task_types(),
        "metric_semantics": allowed_metric_semantics(),
        "ranking_strategies": allowed_ranking_strategies(),
        "scale_strategies": allowed_scale_strategies(),
        "visual_constraints": allowed_visual_constraints(),
    }


def _validate_fields(*, fields: list[str], available_fields: set[str], context: str) -> None:
    missing = sorted(field for field in fields if field not in available_fields)
    if missing:
        raise ValueError(
            f"AnalysisPlan contains fields absent from DataProfile in {context}: {missing}. "
            f"Available fields: {sorted(available_fields)}"
        )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
