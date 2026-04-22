from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.domain.enums import ChartCaseType
from src.domain.models import (
    AnalysisRubric,
    ArtifactBundle,
    CandidateSpecSet,
    ChartReadResult,
    CodeRunResult,
    CodegenResult,
    DataPreparationResult,
    DataProfile,
    EmptyChartCheckResult,
    EvaluationSummaryResult,
    ExecutionPolicy,
    FactExtractionResult,
    PlanningResult,
    QueryIntentBundle,
    QueryUnderstandingResult,
    ReasoningResult,
    RequestAnalysisResult,
    ScenegraphCheckResult,
    SpecValidationResult,
    StructuralSpecMetric,
    ValidationPolicy,
    VegaLiteSpecArtifact,
    VerificationResult,
    VisualFactExtractionResult,
    VisualQualityMetric,
    VisRAGResult,
    VLMAnalysisResult,
)


class PipelineRequest(BaseModel):
    query: str
    data_path: str
    user_context: dict[str, Any] = Field(default_factory=dict)
    run_id: str = Field(default_factory=lambda: uuid4().hex)


class PipelineResult(BaseModel):
    run_id: str
    query: str
    data_path: str
    case_type: ChartCaseType | None = None
    query_understanding: QueryUnderstandingResult
    planning: PlanningResult
    data_profile: DataProfile
    data_preparation: DataPreparationResult
    visrag: VisRAGResult
    codegen: CodegenResult
    execution: CodeRunResult
    artifact_bundle: ArtifactBundle
    chart_read: ChartReadResult
    facts: FactExtractionResult
    reasoning: ReasoningResult
    verification: VerificationResult

    # Migration-friendly optional fields for the new architecture.
    query_intent_bundle: QueryIntentBundle | None = None
    request_analysis: RequestAnalysisResult | None = None
    execution_policy: ExecutionPolicy | None = None
    validation_policy: ValidationPolicy | None = None
    analysis_rubric: AnalysisRubric | None = None
    candidate_spec_set: CandidateSpecSet | None = None
    vega_spec: VegaLiteSpecArtifact | None = None
    spec_validation: SpecValidationResult | None = None
    scenegraph_check: ScenegraphCheckResult | None = None
    empty_chart_check: EmptyChartCheckResult | None = None
    vlm_analysis: VLMAnalysisResult | None = None
    visual_facts: VisualFactExtractionResult | None = None
    structural_spec_metric: StructuralSpecMetric | None = None
    visual_quality_metric: VisualQualityMetric | None = None
    evaluation_summary: EvaluationSummaryResult | None = None
