from typing import Any

from pydantic import BaseModel, Field

from .enums import ArtifactType, ChartCaseType



class StepLog(BaseModel):
    stage: str
    title: str
    summary: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class QueryVariant(BaseModel):
    kind: str
    text: str
    confidence: float = 0.0
    source: str = "heuristic"


class QueryIntentBundle(BaseModel):
    intent: str
    requested_operations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    task_type: str | None = None
    user_goal: str | None = None
    analysis_goal: str | None = None
    confidence: float = 0.0
    query_variants: list[QueryVariant] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)


class QueryUnderstandingResult(BaseModel):
    intent: str
    requested_operations: list[str] = Field(default_factory=list)
    candidate_charts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    # Deprecated compatibility field. New architecture should avoid branching on case type.
    case_type: ChartCaseType | None = None
    confidence: float = 0.0
    task_type: str | None = None
    user_goal: str | None = None
    analysis_goal: str | None = None
    query_variants: list[QueryVariant] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)

    def to_intent_bundle(self) -> QueryIntentBundle:
        return QueryIntentBundle(
            intent=self.intent,
            requested_operations=list(self.requested_operations),
            constraints=list(self.constraints),
            task_type=self.task_type,
            user_goal=self.user_goal,
            analysis_goal=self.analysis_goal,
            confidence=self.confidence,
            query_variants=list(self.query_variants),
            ambiguity_notes=list(self.ambiguity_notes),
        )


class PlanningStep(BaseModel):
    name: str
    description: str


class ExecutionPolicy(BaseModel):
    max_retries: int = 1
    fallback_enabled: bool = True
    retry_strategy: str = "repair_then_fallback"
    prefer_best_ranked_spec: bool = True


class ValidationPolicy(BaseModel):
    use_spec_validator: bool = True
    use_scenegraph_check: bool = True
    use_empty_chart_check: bool = True
    fail_fast_on_schema_error: bool = False


class AnalysisRubric(BaseModel):
    focus_areas: list[str] = Field(default_factory=list)
    output_format: str = "bullet_points"
    strict_visual_only: bool = True
    emphasize_anomalies: bool = True


class PlanningResult(BaseModel):
    # Deprecated compatibility field. New architecture should use policies/rubrics.
    mode: ChartCaseType | None = None
    steps: list[PlanningStep] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    execution_policy: ExecutionPolicy | None = None
    validation_policy: ValidationPolicy | None = None
    analysis_rubric: AnalysisRubric | None = None


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
    typed_columns: list[str] = Field(default_factory=list)
    field_roles: dict[str, str] = Field(default_factory=dict)
    schema_hints: list[str] = Field(default_factory=list)
    complexity_hints: list[str] = Field(default_factory=list)
    cleaning_hints: list[str] = Field(default_factory=list)
    data_complexity: str | None = None


class RequestFieldMapping(BaseModel):
    query_term: str
    column_name: str
    confidence: float = 0.0
    rationale: str = ""


class RequestAnalysisResult(BaseModel):
    grounded_fields: list[str] = Field(default_factory=list)
    ambiguity_report: list[str] = Field(default_factory=list)
    selected_fields: list[str] = Field(default_factory=list)
    normalization_hints: list[str] = Field(default_factory=list)
    mappings: list[RequestFieldMapping] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = 0.0


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


class CandidateSpec(BaseModel):
    spec_id: str
    chart_family: str
    summary: str
    score: float = 0.0
    rationale: str = ""
    visualization_plan: VisualizationPlan | None = None


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


class VegaLiteSpecArtifact(BaseModel):
    spec_json: dict[str, Any] = Field(default_factory=dict)
    version: str | None = None


class SpecValidationResult(BaseModel):
    validated_spec: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    repair_hints: list[str] = Field(default_factory=list)
    is_valid: bool = False


class ScenegraphCheckResult(BaseModel):
    has_marks: bool = False
    has_axes: bool = False
    has_legends: bool = False
    notes: list[str] = Field(default_factory=list)


class EmptyChartCheckResult(BaseModel):
    empty_chart_signal: bool = False
    fallback_request: str | None = None
    non_empty_render: bool = False
    empty_chart_status: str = "unknown"


class PlotImageArtifact(BaseModel):
    image_path: str
    width: int = 0
    height: int = 0


class PlotRenderingResult(BaseModel):
    plot_image: PlotImageArtifact
    rendered_scenegraph: dict[str, Any] = Field(default_factory=dict)
    render_notes: list[str] = Field(default_factory=list)


class VLMAnalysisResult(BaseModel):
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


class InsightVerificationResult(BaseModel):
    verified_insights: list[str] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    insight_verification_summary: str = ""
    all_verified: bool = False


class InsightsResult(BaseModel):
    final_insights: list[str] = Field(default_factory=list)


class StructuralSpecMetric(BaseModel):
    score: float = 0.0
    details: list[str] = Field(default_factory=list)


class VisualQualityMetric(BaseModel):
    score: float = 0.0
    details: list[str] = Field(default_factory=list)


class EvaluationSummaryResult(BaseModel):
    structural_spec_metric: float = 0.0
    visual_quality_metric: float = 0.0
    empty_chart_status: str = "unknown"
    insight_verification_summary: str = ""
    benchmark_report: dict[str, Any] = Field(default_factory=dict)


class CodegenResult(BaseModel):
    language: str = "python"
    chart_type: str
    code: str
    entrypoint: str = "main"
    prompt_path: str | None = None
    raw_response_path: str | None = None
    generated_logic_path: str | None = None


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
