from typing import Any

from pydantic import BaseModel, Field

from .enums import ArtifactType, ChartCaseType


class QueryUnderstandingResult(BaseModel):
    intent: str
    requested_operations: list[str] = Field(default_factory=list)
    candidate_charts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    case_type: ChartCaseType
    confidence: float = 0.0


class PlanningStep(BaseModel):
    name: str
    description: str


class PlanningResult(BaseModel):
    mode: ChartCaseType
    steps: list[PlanningStep] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class DataColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_ratio: float
    unique_count: int


class DataProfile(BaseModel):
    row_count: int
    col_count: int
    columns: list[DataColumnProfile]
    likely_numeric_columns: list[str] = Field(default_factory=list)
    likely_categorical_columns: list[str] = Field(default_factory=list)
    likely_time_columns: list[str] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)


class DataPreparationResult(BaseModel):
    output_path: str
    operations: list[str] = Field(default_factory=list)
    row_count: int
    col_count: int


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


class CodegenResult(BaseModel):
    language: str = "python"
    chart_type: str
    code: str
    entrypoint: str = "main"


class ExecutionMetric(BaseModel):
    name: str
    value: float | int | str
    unit: str = ""


class ArtifactRef(BaseModel):
    artifact_type: ArtifactType
    path: str
    description: str


class ArtifactBundle(BaseModel):
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    manifest_path: str | None = None


class CodeRunResult(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    metrics: list[ExecutionMetric] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    chart_metadata: dict[str, Any] = Field(default_factory=dict)


class ChartElement(BaseModel):
    kind: str
    label: str | None = None
    value: float | int | str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ChartReadResult(BaseModel):
    chart_type: str
    title: str | None = None
    axes: dict[str, str] = Field(default_factory=dict)
    series: list[str] = Field(default_factory=list)
    elements: list[ChartElement] = Field(default_factory=list)
    source_artifact: str | None = None


class Fact(BaseModel):
    name: str
    value: str
    evidence: list[str] = Field(default_factory=list)


class FactExtractionResult(BaseModel):
    facts: list[Fact] = Field(default_factory=list)


class ReasoningStatement(BaseModel):
    text: str
    evidence: list[str] = Field(default_factory=list)


class ReasoningResult(BaseModel):
    summary: str
    statements: list[ReasoningStatement] = Field(default_factory=list)


class VerificationFinding(BaseModel):
    statement: str
    status: str
    evidence: list[str] = Field(default_factory=list)
    notes: str = ""


class VerificationResult(BaseModel):
    all_verified: bool
    findings: list[VerificationFinding] = Field(default_factory=list)
