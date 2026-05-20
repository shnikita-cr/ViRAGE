from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class VLMChartDescriptionResult(BaseModel):
    """PNG-only visual description used by the semantic feedback loop."""

    input_scope: Literal["png_only"] = "png_only"
    visual_description: str = ""
    detected_chart_type: str | None = None
    visible_axes: dict[str, str] = Field(default_factory=dict)
    visible_legend: dict[str, Any] = Field(default_factory=dict)
    visible_labels: list[str] = Field(default_factory=list)
    visible_trends: list[str] = Field(default_factory=list)
    visible_comparisons: list[str] = Field(default_factory=list)
    visible_outliers: list[str] = Field(default_factory=list)
    readability_issues: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ChartFactSummaryResult(BaseModel):
    """Query-free summary of facts visible in the chart description."""

    chart_type: str | None = None
    facts: list[str] = Field(default_factory=list)
    axes: dict[str, str] = Field(default_factory=dict)
    legend: dict[str, Any] = Field(default_factory=dict)
    visible_variables: list[str] = Field(default_factory=list)
    visible_relationships: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)


class ChartAnswerJudgeResult(BaseModel):
    """Semantic check: does the visible chart answer the original request?"""

    answers_user_query: bool = False
    confidence: float = 0.0
    retry_recommendation: Literal["accept", "retry", "reject"] = "retry"
    missing_requirements: list[str] = Field(default_factory=list)
    wrong_or_suspicious_parts: list[str] = Field(default_factory=list)
    improvement_comments: list[str] = Field(default_factory=list)
    feedback_for_next_generation: str = ""


class SemanticChartJudgeResult(BaseModel):
    """Strict one-call VLM judge for chart-grounded semantic feedback."""

    input_scope: Literal["png_query_visual_requirements"] = "png_query_visual_requirements"
    chart_description: str = ""
    detected_chart_type: str | None = None
    visible_axes: dict[str, str] = Field(default_factory=dict)
    visible_legend: dict[str, Any] = Field(default_factory=dict)
    visible_labels: list[str] = Field(default_factory=list)
    visible_fields: list[str] = Field(default_factory=list)
    observed_facts: list[str] = Field(default_factory=list)
    answers_user_query: bool = False
    supports_visible_claims: bool = True
    confidence: float = 0.0
    retry_recommendation: Literal["accept", "retry", "reject"] = "retry"
    missing_requirements: list[str] = Field(default_factory=list)
    wrong_or_suspicious_parts: list[str] = Field(default_factory=list)
    readability_issues: list[str] = Field(default_factory=list)
    improvement_comments: list[str] = Field(default_factory=list)
    feedback_for_next_generation: str = ""
    is_blank_or_unreadable: bool = False
    rationales: dict[str, str] = Field(default_factory=dict)


class ChartGroundedAnalysisRecord(BaseModel):
    record_type: Literal["chart_grounded_analysis"] = "chart_grounded_analysis"
    user_query: str
    used_fields: list[str] = Field(default_factory=list)
    chart_type: str | None = None
    observed_facts: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    feedback_for_next_generation: str = ""
    accepted: bool = False
    retry_recommendation: str = "retry"
    confidence: float = 0.0


class ChartRevisionRecord(BaseModel):
    record_type: Literal["chart_revision_record"] = "chart_revision_record"
    attempt_number: int
    user_query: str
    accepted: bool = False
    retry_recommendation: str = "retry"
    retry_reasons: list[str] = Field(default_factory=list)
    feedback_for_next_generation: str = ""
    generated_spec: dict[str, Any] = Field(default_factory=dict)
    rendered_png_path: str = ""


class VisualFeedbackExample(BaseModel):
    record_type: Literal["visual_feedback"] = "visual_feedback"
    source: str = "virage_semantic_loop"
    status: Literal["rejected_or_needs_improvement", "accepted"] = "rejected_or_needs_improvement"
    created_at: str
    run_id: str
    attempt_number: int
    user_query: str
    user_comment: str = ""
    requested_regeneration: bool = False
    feedback_weight: float = 1.0
    request_analysis_summary: dict[str, Any] = Field(default_factory=dict)
    generated_spec: dict[str, Any] = Field(default_factory=dict)
    rendered_png_path: str = ""
    vlm_chart_description: VLMChartDescriptionResult
    chart_fact_summary: ChartFactSummaryResult
    judge_result: ChartAnswerJudgeResult
    feedback_for_next_generation: str = ""
    rag_usage: dict[str, Any] = Field(default_factory=lambda: {"approved_for_rag": False, "exported_to_rag": False})


class SemanticFeedbackLoopSummary(BaseModel):
    enabled: bool = False
    max_attempts: int = 0
    attempt_count: int = 0
    retry_count: int = 0
    accepted: bool = False
    accepted_attempt: int | None = None
    final_status: Literal["disabled", "accepted", "failed", "skipped"] = "disabled"
    final_confidence: float = 0.0
    saved_feedback_count: int = 0
    missing_requirements: list[str] = Field(default_factory=list)
    improvement_comments: list[str] = Field(default_factory=list)
    feedback_corpus_path: str | None = None
    summary_artifact_path: str | None = None
