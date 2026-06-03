from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, model_validator

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
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _dedupe_fields(self) -> "AnalysisSubtask":
        self.required_fields = _dedupe(self.required_fields)
        self.optional_fields = [field for field in _dedupe(self.optional_fields) if field not in self.required_fields]
        if not self.required_fields:
            raise ValueError(f"Analysis subtask {self.id!r} must contain at least one required field.")
        return self


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
