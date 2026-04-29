from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VisRAGRetrievedExample(BaseModel):
    example_id: str
    source: str
    corpus: str = "unknown"
    chart_type: str
    instruction: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    code_language: str | None = None
    domain: str | None = None
    score: float = 0.0
    rationale: str = ""
    document_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisualizationFieldBinding(BaseModel):
    channel: str
    field_name: str
    field_role: str
    title: str | None = None
    aggregate: str | None = None
    time_unit: str | None = None
    sort: str | None = None
    required: bool = True


class VisualizationTransform(BaseModel):
    kind: str
    field_name: str | None = None
    expression: str | None = None
    aggregate: str | None = None
    group_by: list[str] = Field(default_factory=list)
    order_by: str | None = None
    descending: bool = False
    description: str = ""


class VisualizationAxisInstruction(BaseModel):
    channel: str
    field_name: str
    title: str
    scale_type: str
    format_hint: str | None = None
    rotate_labels: bool = False


class VisualizationPlan(BaseModel):
    chart_family: str
    visual_task: str
    goal: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    field_bindings: list[VisualizationFieldBinding] = Field(default_factory=list)
    transforms: list[VisualizationTransform] = Field(default_factory=list)
    axes: list[VisualizationAxisInstruction] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    build_instructions: list[str] = Field(default_factory=list)
    mark_hints: list[str] = Field(default_factory=list)
    renderer_hints: list[str] = Field(default_factory=list)
    evidence_example_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    vega_lite_ready: bool = True


class CandidateSpec(BaseModel):
    spec_id: str
    chart_family: str
    summary: str
    score: float = 0.0
    rationale: str = ""
    visualization_plan: VisualizationPlan | None = None
    spec_template: dict[str, Any] | None = None
    encoding_roles: dict[str, str] = Field(default_factory=dict)
    transform_types: list[str] = Field(default_factory=list)


class CandidateSpecSet(BaseModel):
    candidate_specs: list[CandidateSpec] = Field(default_factory=list)
    retrieved_examples: list[VisRAGRetrievedExample] = Field(default_factory=list)
    visualization_plan: VisualizationPlan | None = None
    ranking_hints: list[str] = Field(default_factory=list)
    selected_candidate_spec: CandidateSpec | None = None


class VisRAGRecommendation(BaseModel):
    chart_family: str
    rationale: str
    priority: int
    score: float = 0.0
    support_examples: list[str] = Field(default_factory=list)
    instruction_highlights: list[str] = Field(default_factory=list)


class VisRAGResult(BaseModel):
    recommendations: list[VisRAGRecommendation] = Field(default_factory=list)
    visualization_plan: VisualizationPlan | None = None
    rules: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    implementation_notes: list[str] = Field(default_factory=list)
    retrieved_examples: list[VisRAGRetrievedExample] = Field(default_factory=list)
    corpus_status: dict[str, str] = Field(default_factory=dict)
    retrieval_strategy: str = "heuristic_only"
    retrieval_query: str | None = None
    candidate_spec_set: CandidateSpecSet | None = None
