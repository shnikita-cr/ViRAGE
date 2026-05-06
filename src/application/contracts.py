from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.domain.models import (
    AnalysisRubric,
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
    StructuralSpecMetric,
    TokenUsage,
    VegaLiteSpecArtifact,
    VisualFactExtractionResult,
    VisualQualityMetric,
    VisRAGResult,
    VLMAnalysisResult,
)


class PipelineRequest(BaseModel):
    query: str
    data_path: str
    user_context: dict[str, Any] = Field(default_factory=dict)
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_" + uuid4().hex)


class PipelineResult(BaseModel):
    run_id: str
    query: str
    data_path: str

    query_understanding: QueryUnderstandingResult | None = None
    query_intent_bundle: QueryIntentBundle | None = None
    request_analysis: RequestAnalysisResult | None = None
    analysis_rubric: AnalysisRubric | None = None
    data_profile: DataProfile | None = None
    data_preparation: DataPreparationResult | None = None
    visrag: VisRAGResult | None = None
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
    stage_execution_logs: list[StageExecutionLog] = Field(default_factory=list)
    model_call_logs: list[ModelCallLog] = Field(default_factory=list)
    token_usage_summary: TokenUsage = Field(default_factory=TokenUsage)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
