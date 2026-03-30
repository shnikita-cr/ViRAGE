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


class VisRAGRecommendation(BaseModel):
    chart_family: str
    rationale: str
    priority: int


class VisRAGResult(BaseModel):
    recommendations: list[VisRAGRecommendation] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


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
