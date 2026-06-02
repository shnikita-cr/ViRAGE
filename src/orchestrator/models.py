from __future__ import annotations

from typing import Any, Literal

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


class AnalysisSubtask(BaseModel):
    id: str
    task_type: AnalysisTaskType
    query: str
    purpose: str
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)
    priority: int = Field(default=100, ge=1)
    constraints: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""

    @model_validator(mode="after")
    def _dedupe_fields(self) -> "AnalysisSubtask":
        self.required_fields = _dedupe(self.required_fields)
        self.optional_fields = [field for field in _dedupe(self.optional_fields) if field not in self.required_fields]
        return self


class SkippedAnalysisCandidate(BaseModel):
    task_type: AnalysisTaskType
    reason: str
    required_fields: list[str] = Field(default_factory=list)


class AnalysisPlan(BaseModel):
    user_query: str
    data_path: str
    input_type: str = "table"
    original_input_path: str | None = None
    preprocessing_report_path: str | None = None
    max_charts: int = Field(default=3, ge=1, le=3)
    subtasks: list[AnalysisSubtask] = Field(default_factory=list)
    skipped_candidates: list[SkippedAnalysisCandidate] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_subtasks(self) -> "AnalysisPlan":
        if len(self.subtasks) > self.max_charts:
            raise ValueError(f"AnalysisPlan has {len(self.subtasks)} subtasks, max_charts={self.max_charts}.")
        ids = [item.id for item in self.subtasks]
        duplicates = {item for item in ids if ids.count(item) > 1}
        if duplicates:
            raise ValueError(f"Duplicate analysis subtask ids: {sorted(duplicates)}")
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
