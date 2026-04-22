from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.domain.models import DataProfile, QueryUnderstandingResult, RequestAnalysisResult, RequestFieldMapping
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import ainvoke_structured, invoke_structured
from src.services.base import BaseService


class _FieldMappingSchema(BaseModel):
    query_term: str = Field(min_length=1)
    column_name: str = Field(min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    rationale: str = ""


class _RequestAnalysisSchema(BaseModel):
    grounded_fields: list[str] = Field(default_factory=list)
    ambiguity_report: list[str] = Field(default_factory=list)
    selected_fields: list[str] = Field(default_factory=list)
    normalization_hints: list[str] = Field(default_factory=list)
    mappings: list[_FieldMappingSchema] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "selected_fields" not in data and "grounded_fields" in data:
            data["selected_fields"] = list(data["grounded_fields"])
        return data


class RequestAnalyzerService(BaseService):
    def invoke(self, query: str, query_understanding: QueryUnderstandingResult, data_profile: DataProfile, runtime: RuntimeContext) -> RequestAnalysisResult:
        reasoning_llm = runtime.reasoning_llm
        if reasoning_llm is None:
            raise RuntimeError("RequestAnalyzerService requires runtime.reasoning_llm. No reasoning model was provided.")
        parsed = invoke_structured(
            reasoning_llm,
            self._prompt(query, query_understanding, data_profile),
            _RequestAnalysisSchema,
            runtime=runtime,
            stage="request_analyzer",
            role="reasoning",
            examples=[self._example_payload(data_profile)],
            max_attempts=2,
        )
        return self._build_result(parsed)

    async def ainvoke(self, query: str, query_understanding: QueryUnderstandingResult, data_profile: DataProfile, runtime: RuntimeContext) -> RequestAnalysisResult:
        reasoning_llm = runtime.reasoning_llm
        if reasoning_llm is None:
            raise RuntimeError("RequestAnalyzerService requires runtime.reasoning_llm. No reasoning model was provided.")
        parsed = await ainvoke_structured(
            reasoning_llm,
            self._prompt(query, query_understanding, data_profile),
            _RequestAnalysisSchema,
            runtime=runtime,
            stage="request_analyzer",
            role="reasoning",
            examples=[self._example_payload(data_profile)],
            max_attempts=2,
        )
        return self._build_result(parsed)

    def _prompt(self, query: str, query_understanding: QueryUnderstandingResult, data_profile: DataProfile) -> str:
        column_lines = []
        for column in data_profile.columns:
            role = data_profile.field_roles.get(column.name, "unknown")
            column_lines.append(
                f"- {column.name} | dtype={column.dtype} | role={role} | missing_ratio={column.missing_ratio:.3f}"
            )
        profile_lines = "\n".join(column_lines) or "- none"
        variants = "\n".join(f"- {variant.kind}: {variant.text}" for variant in query_understanding.query_variants) or "- none"
        return (
            "You ground a visualization request to real dataset fields.\n"
            "Use the exact schema field names required by the output schema.\n"
            "Select fields directly relevant to the request and report ambiguity explicitly.\n"
            f"User request:\n{query}\n"
            f"Intent: {query_understanding.intent}\n"
            f"Requested operations: {', '.join(query_understanding.requested_operations)}\n"
            f"Constraints: {', '.join(query_understanding.constraints)}\n"
            f"Query variants:\n{variants}\n"
            f"Columns:\n{profile_lines}\n"
            f"Schema hints: {', '.join(data_profile.schema_hints)}\n"
        )

    def _build_result(self, parsed: _RequestAnalysisSchema) -> RequestAnalysisResult:
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

    @staticmethod
    def _example_payload(data_profile: DataProfile) -> dict[str, Any]:
        columns = [column.name for column in data_profile.columns[:2]] or ["x", "y"]
        grounded = columns[:]
        return {
            "grounded_fields": grounded,
            "ambiguity_report": [],
            "selected_fields": grounded,
            "normalization_hints": [],
            "mappings": [
                {
                    "query_term": columns[0],
                    "column_name": columns[0],
                    "confidence": 0.8,
                    "rationale": "schema-grounded example",
                }
            ],
            "missing_fields": [],
            "confidence": 0.75,
        }
