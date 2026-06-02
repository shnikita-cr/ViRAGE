from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, model_validator

from src.domain.models import (
    DataProfile,
    FieldBinding,
    QueryAmbiguity,
    QueryRequestAnalysisResult,
    QueryVariant,
    RequestFieldMapping,
)
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured
from src.services.base import BaseService
from src.services.data_profile_prompt_formatter import DataProfilePromptFormatter

_ALLOWED_VARIANT_KINDS = {
    "canonical",
    "chart_pattern_retrieval",
    "repair_rule_retrieval",
    "analysis_rule_retrieval",
}
_VARIANT_KIND_ALIASES = {
    "spec_retrieval": "chart_pattern_retrieval",
    "schema_grounding": "chart_pattern_retrieval",
    "rag": "chart_pattern_retrieval",
    "retrieval": "chart_pattern_retrieval",
    "analysis": "analysis_rule_retrieval",
    "repair": "repair_rule_retrieval",
}


class _QueryVariantSchema(BaseModel):
    kind: str = "canonical"
    text: str = Field(min_length=1)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"kind": "canonical", "text": value, "confidence": 0.6}
        if isinstance(value, dict):
            data = dict(value)
            if "text" not in data:
                for key in ("query", "variant", "value"):
                    if isinstance(data.get(key), str):
                        data["text"] = data[key]
                        break
            data.setdefault("kind", "canonical")
            data.setdefault("confidence", 0.6)
            return data
        return value


class _FieldMappingSchema(BaseModel):
    query_term: str = Field(min_length=1)
    column_name: str = Field(min_length=1)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    rationale: str = ""


class _FieldBindingSchema(BaseModel):
    field: str = Field(min_length=1)
    role: str = "unspecified"
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    rationale: str = ""


class _AmbiguitySchema(BaseModel):
    missing_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class _QueryRequestAnalysisSchema(BaseModel):
    normalized_query: str = Field(
        min_length=1,
        validation_alias=AliasChoices("normalized_query", "canonical_query", "intent", "analytic_intent", "user_intent"),
    )
    analysis_task: str = "descriptive_analytics"
    selected_fields: list[str] = Field(default_factory=list)
    field_bindings: dict[str, _FieldBindingSchema] = Field(default_factory=dict)
    field_mappings: list[_FieldMappingSchema] = Field(default_factory=list, validation_alias=AliasChoices("field_mappings", "mappings"))
    aggregation_plan: dict[str, Any] = Field(default_factory=dict)
    visual_judge_requirements: dict[str, Any] = Field(default_factory=dict)
    query_variants: list[_QueryVariantSchema] = Field(default_factory=list)
    chart_answerability: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    ambiguity: _AmbiguitySchema = Field(default_factory=_AmbiguitySchema)
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "normalized_query" not in data:
            for key in ("canonical_query", "intent", "analytic_intent", "user_intent"):
                if isinstance(data.get(key), str) and data[key].strip():
                    data["normalized_query"] = data[key]
                    break
        if "field_mappings" not in data and "mappings" in data:
            data["field_mappings"] = data["mappings"]
        if "selected_fields" not in data and "grounded_fields" in data:
            data["selected_fields"] = data.get("grounded_fields") or []
        if "ambiguity" not in data:
            data["ambiguity"] = {
                "missing_fields": data.get("missing_fields") or [],
                "notes": [*(data.get("ambiguity_notes") or []), *(data.get("ambiguity_report") or [])],
                "confidence": data.get("confidence", 0.65),
            }
        data.setdefault("confidence", 0.65)
        return data


class QueryRequestAnalyzerService(BaseService):
    def invoke(
            self,
            query: str,
            user_context: dict[str, Any],
            data_profile: DataProfile,
            runtime: RuntimeContext,
    ) -> QueryRequestAnalysisResult:
        if runtime.reasoning_llm is None:
            raise RuntimeError("QueryRequestAnalyzerService requires runtime.reasoning_llm.")
        parsed = invoke_structured(
            runtime.reasoning_llm,
            self._prompt(query, user_context, data_profile),
            _QueryRequestAnalysisSchema,
            runtime=runtime,
            stage="query_request_analysis",
            role="reasoning",
            examples=[self._example_payload(data_profile)],
            max_attempts=2,
        )
        return self._build_result(parsed, query)

    def _prompt(self, query: str, user_context: dict[str, Any], data_profile: DataProfile) -> str:
        context_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(user_context.items())) or "- none"
        return (
            "You analyze one NL2VIS/data-visual-analysis request and ground it to the real dataset schema.\n"
            "Return one strict JSON object matching the schema. Use exact original field names for selected_fields, "
            "field_bindings.*.field, field_mappings.column_name, and ambiguity.missing_fields. Do not invent fields.\n"
            "Do not generate Vega-Lite. Do not create rag_queries. Do not include generic chart-quality boilerplate.\n"
            "The result must describe only the user intent: analysis_task, selected_fields, "
            "field_bindings, aggregation_plan, visual_judge_requirements, query_variants, chart_answerability, assumptions, and ambiguity. "
            "Do not choose or recommend a chart type here. Chart selection is handled later by RAG and spec generation.\n"
            "visual_judge_requirements must contain only criteria that can be checked from a static PNG chart. "
            "Tooltip-only information is not visible.\n"
            "query_variants are only for retrieval/debug. Prefer kinds: canonical, chart_pattern_retrieval, repair_rule_retrieval, analysis_rule_retrieval.\n"
            "chart_answerability.status must be one of: answerable_by_chart, requires_computation, uncertain.\n\n"
            f"User request:\n{query}\n\n"
            f"User context:\n{context_lines}\n\n"
            f"Dataset profile:\n{DataProfilePromptFormatter.for_query_analysis(data_profile)}\n"
        )

    def _build_result(self, parsed: _QueryRequestAnalysisSchema, original_query: str) -> QueryRequestAnalysisResult:
        selected_fields = self._dedupe(parsed.selected_fields)
        ambiguity = QueryAmbiguity(
            missing_fields=self._dedupe(parsed.ambiguity.missing_fields),
            notes=self._dedupe(parsed.ambiguity.notes),
            confidence=parsed.ambiguity.confidence,
        )
        return QueryRequestAnalysisResult(
            normalized_query=parsed.normalized_query.strip(),
            analysis_task=parsed.analysis_task.strip() or "descriptive_analytics",
            selected_fields=selected_fields,
            field_bindings={
                key.strip(): FieldBinding(**value.model_dump())
                for key, value in parsed.field_bindings.items()
                if key.strip()
            },
            field_mappings=[RequestFieldMapping(**item.model_dump()) for item in parsed.field_mappings],
            aggregation_plan=dict(parsed.aggregation_plan or {}),
            visual_judge_requirements=self._normalize_visual_judge_requirements(parsed.visual_judge_requirements),
            query_variants=self._normalize_variants(parsed.query_variants, original_query),
            chart_answerability=self._normalize_chart_answerability(parsed.chart_answerability),
            assumptions=self._dedupe(parsed.assumptions),
            ambiguity=ambiguity,
            confidence=parsed.confidence,
        )

    @staticmethod
    def _normalize_visual_judge_requirements(raw: dict[str, Any]) -> dict[str, Any]:
        data = dict(raw or {})
        must_be_visible = QueryRequestAnalyzerService._dedupe(
            [str(item) for item in data.get("must_be_visible", []) if str(item).strip()]
        )
        critical_failures = QueryRequestAnalyzerService._dedupe(
            [str(item) for item in data.get("critical_failures", []) if str(item).strip()]
        )
        yes_no_questions = QueryRequestAnalyzerService._dedupe(
            [str(item) for item in data.get("yes_no_questions", []) if str(item).strip()]
        )
        return {
            "must_be_visible": must_be_visible,
            "acceptable_visual_encodings": data.get("acceptable_visual_encodings", {}),
            "critical_failures": critical_failures,
            "yes_no_questions": yes_no_questions[:12],
        }

    @staticmethod
    def _normalize_chart_answerability(value: dict[str, Any]) -> dict[str, Any]:
        data = dict(value or {})
        status = str(data.get("status") or "uncertain").strip().lower()
        if status not in {"answerable_by_chart", "requires_computation", "uncertain"}:
            status = "uncertain"
        reason = str(data.get("reason") or "").strip()
        return {"status": status, "reason": reason}

    def _normalize_variants(self, values: list[_QueryVariantSchema], original_query: str) -> list[QueryVariant]:
        result: list[QueryVariant] = []
        seen_texts: set[str] = set()
        for item in values:
            kind = self._normalize_variant_kind(item.kind)
            text = item.text.strip()
            key = text.lower()
            if text and key not in seen_texts:
                seen_texts.add(key)
                result.append(QueryVariant(kind=kind, text=text, confidence=item.confidence, source="llm"))
        if original_query.strip().lower() not in seen_texts:
            result.insert(0, QueryVariant(kind="canonical", text=original_query.strip(), confidence=0.55, source="derived"))
        if not any(item.kind == "canonical" for item in result):
            result.insert(0, QueryVariant(kind="canonical", text=original_query.strip(), confidence=0.55, source="derived"))
        return result

    @staticmethod
    def _normalize_variant_kind(value: str) -> str:
        kind = str(value or "canonical").strip().lower().replace("-", "_")
        kind = _VARIANT_KIND_ALIASES.get(kind, kind)
        if kind not in _ALLOWED_VARIANT_KINDS:
            kind = "chart_pattern_retrieval"
        return kind

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                result.append(text)
        return result

    @staticmethod
    def _example_payload(data_profile: DataProfile) -> dict[str, Any]:
        columns = [column.name for column in data_profile.columns[:3]] or ["category", "value"]
        x_field = columns[0]
        y_field = columns[1] if len(columns) > 1 else columns[0]
        return {
            "normalized_query": "Show average value over time by category.",
            "analysis_task": "trend",
            "selected_fields": columns[:3],
            "field_bindings": {
                "x": {"field": x_field, "role": "temporal_axis", "confidence": 0.7, "rationale": "example"},
                "y": {"field": y_field, "role": "measure_axis", "confidence": 0.7, "rationale": "example"},
            },
            "field_mappings": [
                {"query_term": x_field, "column_name": x_field, "confidence": 0.8, "rationale": "schema-grounded example"}
            ],
            "aggregation_plan": {"operation": "mean", "column": y_field, "group_by": [x_field]},
            "visual_judge_requirements": {
                "must_be_visible": ["The x-axis and y-axis show the requested fields."],
                "acceptable_visual_encodings": {},
                "critical_failures": ["A required field is not visible in the static chart."],
                "yes_no_questions": ["Does the chart visibly answer the user request?"],
            },
            "query_variants": [
                {"kind": "canonical", "text": "Show average value over time by category.", "confidence": 0.8},
                {"kind": "chart_pattern_retrieval", "text": "time trend mean measure by category", "confidence": 0.7},
            ],
            "chart_answerability": {"status": "answerable_by_chart", "reason": "A static trend chart can answer this request."},
            "assumptions": [],
            "ambiguity": {"missing_fields": [], "notes": [], "confidence": 0.0},
            "confidence": 0.75,
        }
