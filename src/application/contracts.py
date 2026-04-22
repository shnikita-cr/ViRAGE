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
    InsightReasoningResult,
    InsightVerificationResult,
    InsightsResult,
    ModelCallLog,
    PlanningResult,
    PlotRenderingResult,
    QueryIntentBundle,
    QueryUnderstandingResult,
    ReasoningResult,
    RequestAnalysisResult,
    ScenegraphCheckResult,
    SpecValidationResult,
    StepLog,
    StructuralSpecMetric,
    TokenUsage,
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

    query_understanding: QueryUnderstandingResult | None = None
    planning: PlanningResult | None = None
    data_profile: DataProfile | None = None
    data_preparation: DataPreparationResult | None = None
    visrag: VisRAGResult | None = None

    codegen: CodegenResult | None = None
    execution: CodeRunResult | None = None
    artifact_bundle: ArtifactBundle | None = None
    chart_read: ChartReadResult | None = None
    facts: FactExtractionResult | None = None
    reasoning: ReasoningResult | None = None
    verification: VerificationResult | None = None

    query_intent_bundle: QueryIntentBundle | None = None
    request_analysis: RequestAnalysisResult | None = None
    execution_policy: ExecutionPolicy | None = None
    validation_policy: ValidationPolicy | None = None
    analysis_rubric: AnalysisRubric | None = None
    candidate_spec_set: CandidateSpecSet | None = None
    vega_spec: VegaLiteSpecArtifact | None = None
    spec_validation: SpecValidationResult | None = None
    plot_rendering: PlotRenderingResult | None = None
    scenegraph_check: ScenegraphCheckResult | None = None
    empty_chart_check: EmptyChartCheckResult | None = None
    plot_image: dict[str, Any] | None = None
    vlm_analysis: VLMAnalysisResult | None = None
    visual_facts: VisualFactExtractionResult | None = None
    insight_reasoning: InsightReasoningResult | None = None
    insight_verification: InsightVerificationResult | None = None
    insights: InsightsResult | None = None
    structural_spec_metric: StructuralSpecMetric | None = None
    visual_quality_metric: VisualQualityMetric | None = None
    evaluation_summary: EvaluationSummaryResult | None = None
    step_logs: list[StepLog] = Field(default_factory=list)
    model_call_logs: list[ModelCallLog] = Field(default_factory=list)
    token_usage_summary: TokenUsage = Field(default_factory=TokenUsage)
