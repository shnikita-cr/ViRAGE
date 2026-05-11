from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StructuralSpecMetric(BaseModel):
    score: float = 0.0
    mark_score: float = 0.0
    encoding_score: float = 0.0
    transform_score: float = 0.0
    task_alignment_score: float = 0.0
    validity_score: float = 0.0
    empty_chart_penalty: float = 1.0
    encoding_precision: float = 0.0
    encoding_recall: float = 0.0
    transform_precision: float = 0.0
    transform_recall: float = 0.0
    mark_precision: float = 0.0
    mark_recall: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)
    details: list[str] = Field(default_factory=list)


class VisualQualityMetric(BaseModel):
    score: float = 0.0
    prompt_compliance: float = 0.0
    readability: float = 0.0
    insight_supportiveness: float = 0.0
    visualization_type: float = 0.0
    data_encoding: float = 0.0
    data_transformation: float = 0.0
    aesthetics: float = 0.0
    is_blank: bool = False
    weights: dict[str, float] = Field(default_factory=dict)
    rationales: dict[str, str] = Field(default_factory=dict)
    details: list[str] = Field(default_factory=list)


class EvaluationSummaryResult(BaseModel):
    structural_spec_metric: float | None = None
    visual_quality_metric: float | None = None
    empty_chart_status: str = "unknown"
    visualization_error_rate_item: bool | None = None
    empty_chart_rate_item: bool | None = None
    insight_summary: str = ""
    benchmark_report: dict[str, Any] = Field(default_factory=dict)
