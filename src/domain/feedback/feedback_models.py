from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

FeedbackSourceKind = Literal["manual_feedback", "vlm_feedback"]
FeedbackSeverity = Literal["low", "medium", "high"]


class NormalizedFeedbackRecord(BaseModel):
    """Reviewed intermediate feedback record prepared for optional RAG export.

    Raw feedback remains the immutable source of truth. This model stores a
    structured, RAG-facing version of one raw feedback example. Records are not
    retrieved at runtime until approved_for_rag is explicitly true and the export
    script writes them as VisRAG guidance chunks.
    """

    record_type: Literal["normalized_feedback"] = "normalized_feedback"
    feedback_id: str
    source_record_id: str = ""
    source_kind: FeedbackSourceKind = "vlm_feedback"
    source: str = ""
    created_at: str = ""
    run_id: str = ""
    attempt_number: int = 1
    user_query: str = ""
    feedback_type: str = "general_visual_feedback"
    severity: FeedbackSeverity = "medium"
    task_type: str = ""
    chart_family: str = ""
    problem: str = ""
    recommendation: str = ""
    avoid: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)
    fields_used: list[str] = Field(default_factory=list)
    data_profile_summary: dict[str, Any] = Field(default_factory=dict)
    priority: float = Field(default=1.0, ge=0.0, le=4.0)
    approved_for_rag: bool = False
    approval_notes: str = "manual_review_required"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _strip_text(self) -> "NormalizedFeedbackRecord":
        self.feedback_type = self.feedback_type.strip() or "general_visual_feedback"
        self.task_type = self.task_type.strip()
        self.chart_family = self.chart_family.strip()
        self.problem = self.problem.strip()
        self.recommendation = self.recommendation.strip()
        self.avoid = [item.strip() for item in self.avoid if str(item).strip()]
        self.quality_checks = [item.strip() for item in self.quality_checks if str(item).strip()]
        self.fields_used = list(dict.fromkeys([item.strip() for item in self.fields_used if str(item).strip()]))
        return self

    def to_chunk_text(self) -> str:
        sections = [
            f"Feedback type: {self.feedback_type}",
            f"Severity: {self.severity}",
        ]
        if self.task_type:
            sections.append(f"Analytical task: {self.task_type}")
        if self.chart_family:
            sections.append(f"Chart family: {self.chart_family}")
        if self.fields_used:
            sections.append("Fields used: " + ", ".join(self.fields_used))
        if self.problem:
            sections.append("Problem: " + self.problem)
        if self.recommendation:
            sections.append("Recommendation: " + self.recommendation)
        if self.avoid:
            sections.append("Avoid:\n" + "\n".join(f"- {item}" for item in self.avoid))
        if self.quality_checks:
            sections.append("Quality checks:\n" + "\n".join(f"- {item}" for item in self.quality_checks))
        return "\n\n".join(sections).strip()
