from typing import Any

from typing_extensions import NotRequired, TypedDict

from src.domain.enums import PipelineStage
from src.domain.models import (
    AnalysisRubric,
    ChartAnswerJudgeResult,
    ChartFactSummaryResult,
    CandidateSpecSet,
    DataPreparationResult,
    DataProfile,
    EmptyChartCheckResult,
    EvaluationSummaryResult,
    InsightReasoningResult,
    InsightVerificationResult,
    InsightsResult,
    ModelCallLog,
    PlotRenderingResult,
    QueryIntentBundle,
    QueryUnderstandingResult,
    RequestAnalysisResult,
    ScenegraphCheckResult,
    SpecValidationResult,
    StepLog,
    StageExecutionLog,
    SemanticFeedbackLoopSummary,
    StructuralSpecMetric,
    TokenUsage,
    VegaLiteSpecArtifact,
    VisualFactExtractionResult,
    VisualQualityMetric,
    VisRAGResult,
    VLMAnalysisResult,
    VLMChartDescriptionResult,
    VisualFeedbackExample,
)


class PipelineState(TypedDict, total=False):
    run_id: str
    query: str
    data_path: str
    user_context: dict[str, Any]
    stage: PipelineStage
    trace: list[str]
    errors: list[str]
    artifact_paths: dict[str, str]

    technical_attempt_number: NotRequired[int]
    technical_retry_feedback: NotRequired[dict[str, Any]]
    technical_status: NotRequired[str]

    semantic_attempt_number: NotRequired[int]
    semantic_feedback_items: NotRequired[list[str]]
    semantic_chart_fact_history: NotRequired[list[dict[str, Any]]]
    semantic_status: NotRequired[str]
    semantic_retry_feedback: NotRequired[str]

    query_understanding: NotRequired[QueryUnderstandingResult]
    query_intent_bundle: NotRequired[QueryIntentBundle]
    request_analysis: NotRequired[RequestAnalysisResult]
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
    vlm_chart_description: NotRequired[VLMChartDescriptionResult]
    chart_fact_summary: NotRequired[ChartFactSummaryResult]
    chart_answer_judge: NotRequired[ChartAnswerJudgeResult]
    visual_feedback_examples: NotRequired[list[VisualFeedbackExample]]
    semantic_feedback_loop_summary: NotRequired[SemanticFeedbackLoopSummary]
    vlm_analysis: NotRequired[VLMAnalysisResult]
    visual_facts: NotRequired[VisualFactExtractionResult]
    insight_reasoning: NotRequired[InsightReasoningResult]
    insight_verification: NotRequired[InsightVerificationResult]
    insights: NotRequired[InsightsResult]
    structural_spec_metric: NotRequired[StructuralSpecMetric]
    visual_quality_metric: NotRequired[VisualQualityMetric]
    evaluation_summary: NotRequired[EvaluationSummaryResult]

    step_logs: NotRequired[list[StepLog]]
    stage_execution_logs: NotRequired[list[StageExecutionLog]]
    model_call_logs: NotRequired[list[ModelCallLog]]
    token_usage_summary: NotRequired[TokenUsage]
