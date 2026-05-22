from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.domain.data_models import DataPreparationResult, DataProfile, QueryRequestAnalysisResult
from src.domain.visrag_models import VisRAGResult

SpecGenerationBackendName = Literal["vegachat_codegen"]


class SpecGenerationRequest(BaseModel):
    query: str
    prepared: DataPreparationResult
    data_profile: DataProfile | None = None
    query_request_analysis: QueryRequestAnalysisResult | None = None
    visrag: VisRAGResult | None = None
    generation_attempt_number: int = Field(default=1, ge=1)
    max_generation_attempts: int = Field(default=1, ge=1)
    previous_validation_errors: list[str] = Field(default_factory=list)
    previous_repair_hints: list[str] = Field(default_factory=list)
    previous_invalid_spec: dict[str, Any] | None = None
    previous_semantic_feedback: list[str] = Field(default_factory=list)
    previous_chart_facts: list[dict[str, Any]] = Field(default_factory=list)
    visual_judge_requirements: dict[str, Any] = Field(default_factory=dict)


class SpecGenerationAttempt(BaseModel):
    attempt_number: int
    status: str
    raw_response: str = ""
    explanation: str | None = None
    spec_json: dict[str, Any] | None = None
    error: str | None = None


class SpecGenerationResult(BaseModel):
    backend_name: str
    prompt_version: str | None = None
    spec_json: dict[str, Any] = Field(default_factory=dict)
    spec_without_runtime_data: dict[str, Any] = Field(default_factory=dict)
    explanation: str | None = None
    attempts: list[SpecGenerationAttempt] = Field(default_factory=list)
    warning_messages: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    used_visrag_context: bool = False
    generation_attempt_number: int = Field(default=1, ge=1)
    max_generation_attempts: int = Field(default=1, ge=1)
    previous_validation_errors: list[str] = Field(default_factory=list)
    previous_repair_hints: list[str] = Field(default_factory=list)
    previous_semantic_feedback: list[str] = Field(default_factory=list)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)
