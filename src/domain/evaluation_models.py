from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StructuralSpecMetric(BaseModel):
    score: float = 0.0
    mark_score: float = 0.0
    encoding_score: float = 0.0
    transform_score: float = 0.0
    task_alignment_score: float = 0.0
    details: list[str] = Field(default_factory=list)


class VisualQualityMetric(BaseModel):
    score: float = 0.0
    prompt_compliance: float = 0.0
    readability: float = 0.0
    insight_supportiveness: float = 0.0
    details: list[str] = Field(default_factory=list)


class EvaluationSummaryResult(BaseModel):
    structural_spec_metric: float | None = None
    visual_quality_metric: float | None = None
    empty_chart_status: str = "unknown"
    insight_verification_summary: str = ""
    benchmark_report: dict[str, Any] = Field(default_factory=dict)
