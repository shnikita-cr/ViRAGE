from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from src.domain.enums import ChartCaseType
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


class PipelineRequest(BaseModel):
    query: str
    data_path: str
    user_context: dict[str, Any] = Field(default_factory=dict)
    run_id: str = Field(default_factory=lambda: uuid4().hex)


class PipelineResult(BaseModel):
    run_id: str
    query: str
    data_path: str
    case_type: ChartCaseType
    query_understanding_result: QueryUnderstandingResult
    planning_result: PlanningResult
    data_profile: DataProfile
    data_preparation_result: DataPreparationResult
    visrag_result: VisRAGResult
    codegen_result: CodegenResult
    execution_result: CodeRunResult
    artifact_bundle: ArtifactBundle
    chart_read_result: ChartReadResult
    extracted_facts: FactExtractionResult
    reasoning_result: ReasoningResult
    verification_result: VerificationResult
