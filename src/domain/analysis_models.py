from __future__ import annotations

from pydantic import BaseModel, Field


class AnalysisRubric(BaseModel):
    focus_areas: list[str] = Field(default_factory=list)
    output_format: str = "bullet_points"
    strict_visual_only: bool = True
    emphasize_anomalies: bool = True


class VLMAnalysisResult(BaseModel):
    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    suggested_followup_questions: list[str] = Field(default_factory=list)
    visual_observations: list[str] = Field(default_factory=list)
    extracted_visual_facts: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class VisualFact(BaseModel):
    name: str
    value: str
    evidence_refs: list[str] = Field(default_factory=list)


class VisualFactExtractionResult(BaseModel):
    visual_facts: list[VisualFact] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class InsightCandidate(BaseModel):
    statement: str
    confidence: float = 0.0
    reasoning_chain: list[str] = Field(default_factory=list)


class InsightReasoningResult(BaseModel):
    insight_candidates: list[InsightCandidate] = Field(default_factory=list)
    reasoning_chain: list[str] = Field(default_factory=list)


class InsightsResult(BaseModel):
    final_insights: list[str] = Field(default_factory=list)
