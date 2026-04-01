from typing import Any

from typing_extensions import NotRequired, TypedDict

from src.domain.enums import ChartCaseType, PipelineStage
from src.domain.models import (
    ArtifactBundle,
    ChartReadResult,
    CodeRunResult,
    CodegenResult,
    DataPreparationResult,
    DataProfile,
    FactExtractionResult,
    PlanningResult,
    QueryUnderstandingResult,
    ReasoningResult,
    VerificationResult,
    VisRAGResult,
)


class PipelineState(TypedDict):
    run_id: str
    query: str
    data_path: str
    user_context: dict[str, Any]
    stage: PipelineStage
    case_type: ChartCaseType
    trace: list[str]
    errors: list[str]
    query_understanding: NotRequired[QueryUnderstandingResult]
    planning: NotRequired[PlanningResult]
    data_profile: NotRequired[DataProfile]
    data_preparation: NotRequired[DataPreparationResult]
    visrag: NotRequired[VisRAGResult]
    codegen: NotRequired[CodegenResult]
    execution: NotRequired[CodeRunResult]
    artifact_bundle: NotRequired[ArtifactBundle]
    chart_read: NotRequired[ChartReadResult]
    facts: NotRequired[FactExtractionResult]
    reasoning: NotRequired[ReasoningResult]
    verification: NotRequired[VerificationResult]
