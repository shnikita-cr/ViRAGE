from typing import Any

from typing_extensions import NotRequired, TypedDict

from src.domain.enums import ChartCaseType, PipelineStage
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
    StepLog,
    ExecutionPolicy,
    InsightReasoningResult,
    InsightVerificationResult,
    InsightsResult,
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
    PlotRenderingResult,
)


class PipelineState(TypedDict):
    run_id: str
    query: str
    data_path: str
    user_context: dict[str, Any]
    stage: PipelineStage
    case_type: ChartCaseType | None
    trace: list[str]
    errors: list[str]
    query_understanding: NotRequired[QueryUnderstandingResult]
    query_intent_bundle: NotRequired[QueryIntentBundle]
    request_analysis: NotRequired[RequestAnalysisResult]
    planning: NotRequired[PlanningResult]
    execution_policy: NotRequired[ExecutionPolicy]
    validation_policy: NotRequired[ValidationPolicy]
    analysis_rubric: NotRequired[AnalysisRubric]
    data_profile: NotRequired[DataProfile]
    data_preparation: NotRequired[DataPreparationResult]
    visrag: NotRequired[VisRAGResult]
    candidate_spec_set: NotRequired[CandidateSpecSet]
    vega_spec: NotRequired[VegaLiteSpecArtifact]
    spec_validation: NotRequired[SpecValidationResult]
    plot_rendering: NotRequired[PlotRenderingResult]
    scenegraph_check: NotRequired[ScenegraphCheckResult]
    empty_chart_check: NotRequired[EmptyChartCheckResult]
    plot_image: NotRequired[dict[str, Any]]
    vlm_analysis: NotRequired[VLMAnalysisResult]
    visual_facts: NotRequired[VisualFactExtractionResult]
    structural_spec_metric: NotRequired[StructuralSpecMetric]
    visual_quality_metric: NotRequired[VisualQualityMetric]
    evaluation_summary: NotRequired[EvaluationSummaryResult]
    step_logs: NotRequired[list[StepLog]]
    codegen: NotRequired[CodegenResult]
    execution: NotRequired[CodeRunResult]
    artifact_bundle: NotRequired[ArtifactBundle]
    chart_read: NotRequired[ChartReadResult]
    facts: NotRequired[FactExtractionResult]
    reasoning: NotRequired[ReasoningResult]
    verification: NotRequired[VerificationResult]
    insight_reasoning: NotRequired[InsightReasoningResult]
    insight_verification: NotRequired[InsightVerificationResult]
    insights: NotRequired[InsightsResult]
