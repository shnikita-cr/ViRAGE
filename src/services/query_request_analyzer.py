from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, model_validator

from src.domain.models import (
    DataProfile,
    QueryUnderstandingResult,
    QueryVariant,
    RequestAnalysisResult,
    RequestFieldMapping,
)
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured
from src.services.base import BaseService


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


class _QueryRequestAnalysisSchema(BaseModel):
    intent: str = Field(min_length=1, validation_alias=AliasChoices("intent", "analytic_intent", "user_intent"))
    requested_operations: list[str] = Field(default_factory=list)
    candidate_charts: list[str] = Field(default_factory=list, validation_alias=AliasChoices("candidate_charts", "likely_chart_families"))
    constraints: list[str] = Field(default_factory=list)
    task_type: str = "descriptive_analytics"
    user_goal: str = "understand the data visually"
    analysis_goal: str = "extract key visual patterns"
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    query_variants: list[_QueryVariantSchema] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)

    grounded_fields: list[str] = Field(default_factory=list)
    ambiguity_report: list[str] = Field(default_factory=list)
    selected_fields: list[str] = Field(default_factory=list)
    normalization_hints: list[str] = Field(default_factory=list)
    mappings: list[_FieldMappingSchema] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    field_roles: dict[str, str] = Field(default_factory=dict)
    visual_constraints: list[str] = Field(default_factory=list)
    chart_quality_requirements: list[str] = Field(default_factory=list)
    rag_queries: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "intent" not in data:
            for key in ("analytic_intent", "user_intent"):
                if key in data:
                    data["intent"] = data[key]
                    break
        if "selected_fields" not in data and "grounded_fields" in data:
            data["selected_fields"] = list(data.get("grounded_fields") or [])
        if "candidate_charts" not in data and "likely_chart_families" in data:
            data["candidate_charts"] = data["likely_chart_families"]
        data.setdefault("confidence", 0.65)
        return data


class QueryRequestAnalysisResult(BaseModel):
    query_understanding: QueryUnderstandingResult
    request_analysis: RequestAnalysisResult
    field_roles: dict[str, str] = Field(default_factory=dict)
    visual_constraints: list[str] = Field(default_factory=list)
    chart_quality_requirements: list[str] = Field(default_factory=list)
    rag_queries: list[str] = Field(default_factory=list)


class QueryRequestAnalyzerService(BaseService):
    def invoke(
        self,
        query: str,
        user_context: dict[str, Any],
        data_profile: DataProfile,
        runtime: RuntimeContext,
        compact_data_profile: dict[str, Any] | None = None,
    ) -> QueryRequestAnalysisResult:
        if runtime.reasoning_llm is None:
            raise RuntimeError("QueryRequestAnalyzerService requires runtime.reasoning_llm.")
        parsed = invoke_structured(
            runtime.reasoning_llm,
            self._prompt(query, user_context, data_profile, compact_data_profile),
            _QueryRequestAnalysisSchema,
            runtime=runtime,
            stage="query_request_analysis",
            role="reasoning",
            examples=[self._example_payload(data_profile)],
            max_attempts=2,
        )
        return self._build_result(parsed, query)

    def _prompt(self, query: str, user_context: dict[str, Any], data_profile: DataProfile, compact_data_profile: dict[str, Any] | None) -> str:
        context_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(user_context.items())) or "- none"
        return (
            "You analyze one NL2VIS/data-visual-analysis request and ground it to the real dataset schema.\n"
            "Return one strict JSON object matching the schema. Use exact original field names for selected_fields, "
            "grounded_fields, mappings.column_name, and missing_fields. Do not invent fields.\n"
            "Also return chart quality requirements that the generated chart must satisfy. These requirements must include "
            "readable axis titles, readable labels, required legends, data source/field names, and aggregation names where aggregation is used.\n\n"
            f"User request:\n{query}\n\n"
            f"User context:\n{context_lines}\n\n"
            f"Dataset profile:\n{self._profile_context(data_profile, compact_data_profile)}\n"
        )

    @staticmethod
    def _profile_context(data_profile: DataProfile, compact_data_profile: dict[str, Any] | None) -> str:
        if compact_data_profile:
            lines = [
                f"Rows={compact_data_profile.get('row_count')}; columns={compact_data_profile.get('column_count')}; included={compact_data_profile.get('included_column_count')}"
            ]
            for column in compact_data_profile.get("columns", []):
                if isinstance(column, dict):
                    lines.append(
                        f"- {column.get('original')} | safe={column.get('safe')} | type={column.get('type')} | "
                        f"role={column.get('role')} | missing={column.get('missing_ratio')} | unique={column.get('unique_count')}"
                    )
            notes = compact_data_profile.get("quality_notes_top") or []
            if notes:
                lines.append("Quality notes: " + "; ".join(str(item) for item in notes[:5]))
            return "\n".join(lines)
        lines = [f"Rows={data_profile.row_count}; columns={data_profile.col_count}"]
        for column in data_profile.columns[:30]:
            role = data_profile.field_roles.get(column.name, "unknown")
            lines.append(
                f"- {column.name} | safe={column.safe_name or column.name} | type={column.dtype} | role={role} | "
                f"missing={column.missing_ratio:.3f} | unique={column.unique_count}"
            )
        return "\n".join(lines)

    def _build_result(self, parsed: _QueryRequestAnalysisSchema, original_query: str) -> QueryRequestAnalysisResult:
        understanding = QueryUnderstandingResult(
            intent=parsed.intent.strip(),
            requested_operations=self._dedupe(parsed.requested_operations),
            candidate_charts=self._normalize_chart_names(parsed.candidate_charts),
            constraints=self._dedupe([*parsed.constraints, *parsed.visual_constraints]),
            confidence=parsed.confidence,
            task_type=parsed.task_type.strip(),
            user_goal=parsed.user_goal.strip(),
            analysis_goal=parsed.analysis_goal.strip(),
            query_variants=self._normalize_variants(parsed.query_variants, original_query, parsed.rag_queries),
            ambiguity_notes=self._dedupe([*parsed.ambiguity_notes, *parsed.ambiguity_report]),
        )
        request = RequestAnalysisResult(
            grounded_fields=self._dedupe(parsed.grounded_fields),
            ambiguity_report=self._dedupe(parsed.ambiguity_report),
            selected_fields=self._dedupe(parsed.selected_fields or parsed.grounded_fields),
            normalization_hints=self._dedupe(parsed.normalization_hints),
            mappings=[RequestFieldMapping(**item.model_dump()) for item in parsed.mappings],
            missing_fields=self._dedupe(parsed.missing_fields),
            confidence=parsed.confidence,
        )
        return QueryRequestAnalysisResult(
            query_understanding=understanding,
            request_analysis=request,
            field_roles=dict(parsed.field_roles or {}),
            visual_constraints=self._dedupe(parsed.visual_constraints),
            chart_quality_requirements=self._quality_requirements(parsed.chart_quality_requirements),
            rag_queries=self._dedupe(parsed.rag_queries),
        )

    def _normalize_variants(self, values: list[_QueryVariantSchema], original_query: str, rag_queries: list[str]) -> list[QueryVariant]:
        required = {"canonical", "schema_grounding", "spec_retrieval", "analysis"}
        result: list[QueryVariant] = []
        seen: set[tuple[str, str]] = set()
        for item in values:
            kind = item.kind.strip().lower().replace("-", "_") or "canonical"
            text = item.text.strip()
            key = (kind, text.lower())
            if text and key not in seen:
                seen.add(key)
                result.append(QueryVariant(kind=kind, text=text, confidence=item.confidence, source="llm"))
        for text in rag_queries:
            text = str(text).strip()
            key = ("spec_retrieval", text.lower())
            if text and key not in seen:
                seen.add(key)
                result.append(QueryVariant(kind="spec_retrieval", text=text, confidence=0.7, source="llm"))
        existing = {item.kind for item in result}
        for kind in sorted(required - existing):
            result.append(QueryVariant(kind=kind, text=original_query.strip(), confidence=0.51, source="derived"))
        return result

    @staticmethod
    def _quality_requirements(values: list[str]) -> list[str]:
        base = [
            "Axis titles must be explicit and readable.",
            "Axis titles must name the data source fields and aggregation, for example mean PSNR by Method.",
            "Legends are required whenever color, shape, size, strokeDash, or metric series are encoded.",
            "Category labels must fit the chart or use horizontal bars, rotation, faceting, or larger size.",
            "Multi-metric charts must clearly label every metric and avoid misleading shared scales.",
        ]
        return QueryRequestAnalyzerService._dedupe([*values, *base])

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
    def _normalize_chart_names(values: list[str]) -> list[str]:
        mapping = {
            "bar chart": "bar", "bar plot": "bar", "line chart": "line", "line plot": "line",
            "scatter plot": "scatter", "scatter chart": "scatter", "histogram chart": "histogram",
            "heat map": "heatmap", "box plot": "boxplot", "box-and-whisker": "boxplot",
        }
        return QueryRequestAnalyzerService._dedupe([mapping.get(str(v).strip().lower(), str(v).strip().lower()) for v in values])

    @staticmethod
    def _example_payload(data_profile: DataProfile) -> dict[str, Any]:
        cols = [column.name for column in data_profile.columns[:2]] or ["category", "value"]
        return {
            "intent": "Compare selected fields with an appropriate chart.",
            "requested_operations": ["aggregate:mean"],
            "candidate_charts": ["bar"],
            "constraints": [],
            "task_type": "comparison",
            "user_goal": "compare values",
            "analysis_goal": "show aggregated comparison",
            "confidence": 0.75,
            "query_variants": [{"kind": "canonical", "text": "Compare fields", "confidence": 0.7}],
            "ambiguity_notes": [],
            "grounded_fields": cols,
            "selected_fields": cols,
            "normalization_hints": [],
            "mappings": [{"query_term": cols[0], "column_name": cols[0], "confidence": 0.8, "rationale": "schema-grounded example"}],
            "missing_fields": [],
            "field_roles": {cols[0]: "dimension"},
            "visual_constraints": ["readable_labels_required"],
            "chart_quality_requirements": ["Axis titles include aggregation and source fields."],
            "rag_queries": ["bar chart aggregated comparison"],
        }
