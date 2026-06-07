from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, model_validator

from src.orchestrator.contracts.planning_contract import (
    allowed_metric_semantics,
    allowed_ranking_strategies,
    allowed_scale_strategies,
    allowed_visual_constraints,
    is_problem_ranking_strategy,
    requires_problem_ranking,
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
    metric_semantics: dict[str, Any] = Field(default_factory=dict)
    ranking_strategy: str | None = None
    scale_strategy: str | None = None
    visual_constraints: list[str] = Field(default_factory=list)
    comparison_group_id: str | None = None
    rationale: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _coerce_controlled_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data["ranking_strategy"] = _normalize_controlled_value(
            data.get("ranking_strategy"),
            allowed=set(allowed_ranking_strategies()),
        )
        data["scale_strategy"] = _normalize_controlled_value(
            data.get("scale_strategy"),
            allowed=set(allowed_scale_strategies()),
        )
        data["metric_semantics"] = _normalize_metric_semantics(data.get("metric_semantics") or {})
        return data

    @model_validator(mode="after")
    def _dedupe_fields(self) -> "AnalysisSubtask":
        self.required_fields = _dedupe(self.required_fields)
        self.optional_fields = [field for field in _dedupe(self.optional_fields) if field not in self.required_fields]
        self.visual_constraints = _dedupe_allowed(self.visual_constraints, allowed=set(allowed_visual_constraints()))
        self.metric_semantics = _normalize_metric_semantics(self.metric_semantics)
        self.ranking_strategy = _normalize_controlled_value(self.ranking_strategy, allowed=set(allowed_ranking_strategies()))
        self.scale_strategy = _normalize_controlled_value(self.scale_strategy, allowed=set(allowed_scale_strategies()))
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


def _normalize_metric_semantics(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    allowed = set(allowed_metric_semantics())
    result: dict[str, str] = {}
    for key, raw in value.items():
        field = str(key).strip()
        semantic = _normalize_metric_semantic_value(raw, allowed=allowed)
        if field and semantic:
            result[field] = semantic
    return result


def _normalize_metric_semantic_value(value: Any, *, allowed: set[str]) -> str | None:
    raw: Any = value
    if isinstance(value, dict):
        raw = (
            value.get("metric_semantic")
            or value.get("semantic")
            or value.get("direction")
            or value.get("quality_direction")
            or value.get("value")
        )
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in allowed else None


def _normalize_controlled_value(value: Any, *, allowed: set[str]) -> str | None:
    if isinstance(value, dict):
        value = (
            value.get("ranking_strategy")
            or value.get("scale_strategy")
            or value.get("strategy")
            or value.get("value")
            or value.get("name")
        )
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in allowed else None


def _dedupe_allowed(values: list[Any], *, allowed: set[str]) -> list[str]:
    return [value for value in _dedupe([str(item) for item in values]) if value in allowed]


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
