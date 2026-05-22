from typing import Any

from typing_extensions import NotRequired, TypedDict

from src.domain.enums import PipelineStage
from src.domain.models import (
    AnalysisRubric,
    ChartAnswerJudgeResult,
    ChartFactSummaryResult,
    DataPreparationResult,
    DataProfile,
    EmptyChartCheckResult,
    EvaluationSummaryResult,
    ModelCallLog,
    PlotRenderingResult,
    QueryRequestAnalysisResult,
    ScenegraphCheckResult,
    SpecValidationResult,
    StepLog,
    StageExecutionLog,
    SemanticFeedbackLoopSummary,
    StructuralSpecMetric,
    TokenUsage,
    VegaLiteSpecArtifact,
    VisRAGResult,
    VLMAnalysisResult,
    VLMChartDescriptionResult,
    VisualFeedbackExample,
    SemanticChartJudgeResult,
    VisualChartJudgeResult,
    ChartGroundedAnalysisRecord,
    ChartRevisionRecord,
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
    semantic_retry_reasons: NotRequired[list[str]]

    query_request_analysis: NotRequired[QueryRequestAnalysisResult]
    analysis_rubric: NotRequired[AnalysisRubric]
    data_profile: NotRequired[DataProfile]
    data_preparation: NotRequired[DataPreparationResult]
    visrag: NotRequired[VisRAGResult]
    vega_spec: NotRequired[VegaLiteSpecArtifact]
    spec_validation: NotRequired[SpecValidationResult]
    plot_rendering: NotRequired[PlotRenderingResult]
    scenegraph_check: NotRequired[ScenegraphCheckResult]
    empty_chart_check: NotRequired[EmptyChartCheckResult]
    plot_image: NotRequired[dict[str, Any]]
    vlm_chart_description: NotRequired[VLMChartDescriptionResult]
    chart_fact_summary: NotRequired[ChartFactSummaryResult]
    chart_answer_judge: NotRequired[ChartAnswerJudgeResult]
    visual_chart_judge: NotRequired[VisualChartJudgeResult]
    semantic_chart_judge: NotRequired[SemanticChartJudgeResult]
    chart_analysis: NotRequired[ChartGroundedAnalysisRecord]
    chart_revision_record: NotRequired[ChartRevisionRecord]
    visual_feedback_examples: NotRequired[list[VisualFeedbackExample]]
    semantic_feedback_loop_summary: NotRequired[SemanticFeedbackLoopSummary]
    vlm_analysis: NotRequired[VLMAnalysisResult]
    structural_spec_metric: NotRequired[StructuralSpecMetric]
    evaluation_summary: NotRequired[EvaluationSummaryResult]

    step_logs: NotRequired[list[StepLog]]
    stage_execution_logs: NotRequired[list[StageExecutionLog]]
    model_call_logs: NotRequired[list[ModelCallLog]]
    token_usage_summary: NotRequired[TokenUsage]
