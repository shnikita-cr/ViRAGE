from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import DataProfile, QueryUnderstandingResult, RequestAnalysisResult, RequestFieldMapping
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured
from src.services.base import BaseService


class _FieldMappingSchema(BaseModel):
    query_term: str = Field(min_length=1)
    column_name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class _RequestAnalysisSchema(BaseModel):
    grounded_fields: list[str] = Field(default_factory=list)
    ambiguity_report: list[str] = Field(default_factory=list)
    selected_fields: list[str] = Field(default_factory=list)
    normalization_hints: list[str] = Field(default_factory=list)
    mappings: list[_FieldMappingSchema] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class RequestAnalyzerService(BaseService):
    def invoke(
            self,
            query: str,
            query_understanding: QueryUnderstandingResult,
            data_profile: DataProfile,
            runtime: RuntimeContext,
    ) -> RequestAnalysisResult:
        reasoning_llm = runtime.reasoning_llm
        if reasoning_llm is None:
            raise RuntimeError(
                "RequestAnalyzerService requires runtime.reasoning_llm. No reasoning model was provided.")

        column_lines = []
        for column in data_profile.columns:
            role = data_profile.field_roles.get(column.name, "unknown")
            column_lines.append(
                f"- {column.name} | dtype={column.dtype} | role={role} | missing_ratio={column.missing_ratio:.3f}"
            )
        profile_lines = "\n".join(column_lines) or "- none"
        prompt = (
            "You ground a visualization request to real dataset fields.\n"
            "Return only structured output.\n"
            "Select fields that are directly relevant to the request, report ambiguity, and list missing concepts when needed.\n"
            f"User request:\n{query}\n"
            f"Intent: {query_understanding.intent}\n"
            f"Requested operations: {', '.join(query_understanding.requested_operations)}\n"
            f"Constraints: {', '.join(query_understanding.constraints)}\n"
            f"Columns:\n{profile_lines}\n"
            f"Schema hints: {', '.join(data_profile.schema_hints)}\n"
        )
        parsed = invoke_structured(reasoning_llm, prompt, _RequestAnalysisSchema)
        return RequestAnalysisResult(
            grounded_fields=self._dedupe(parsed.grounded_fields),
            ambiguity_report=self._dedupe(parsed.ambiguity_report),
            selected_fields=self._dedupe(parsed.selected_fields),
            normalization_hints=self._dedupe(parsed.normalization_hints),
            mappings=[RequestFieldMapping(**item.model_dump()) for item in parsed.mappings],
            missing_fields=self._dedupe(parsed.missing_fields),
            confidence=parsed.confidence,
        )

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = value.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        return result
