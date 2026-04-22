from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, model_validator

from src.domain.models import QueryUnderstandingResult, QueryVariant
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import ainvoke_structured, invoke_structured
from src.services.base import BaseService


class _QueryVariantSchema(BaseModel):
    kind: str = "canonical"
    text: str = Field(min_length=1)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _coerce_variant(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"kind": "canonical", "text": value, "confidence": 0.6}
        if isinstance(value, dict):
            data = dict(value)
            if "text" not in data:
                for alt in ("query", "variant", "value"):
                    if isinstance(data.get(alt), str):
                        data["text"] = data[alt]
                        break
            data.setdefault("kind", "canonical")
            data.setdefault("confidence", 0.6)
            return data
        return value


class _QueryUnderstandingSchema(BaseModel):
    intent: str = Field(min_length=1, validation_alias=AliasChoices("intent", "analytic_intent"))
    requested_operations: list[str] = Field(default_factory=list)
    candidate_charts: list[str] = Field(default_factory=list, validation_alias=AliasChoices("candidate_charts", "likely_chart_families"))
    constraints: list[str] = Field(default_factory=list)
    task_type: str = Field(default="descriptive_analytics")
    user_goal: str = Field(default="understand the data visually")
    analysis_goal: str = Field(default="extract key visual patterns")
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    query_variants: list[_QueryVariantSchema] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "intent" not in data and "analytic_intent" in data:
            data["intent"] = data["analytic_intent"]
        if "candidate_charts" not in data and "likely_chart_families" in data:
            data["candidate_charts"] = data["likely_chart_families"]
        if "query_variants" in data and isinstance(data["query_variants"], list):
            data["query_variants"] = [
                {"kind": "canonical", "text": item, "confidence": 0.6} if isinstance(item, str) else item
                for item in data["query_variants"]
            ]
        data.setdefault("confidence", 0.65)
        return data


class QueryUnderstandingService(BaseService):
    def invoke(self, query: str, user_context: dict[str, Any], runtime: RuntimeContext) -> QueryUnderstandingResult:
        reasoning_llm = runtime.reasoning_llm
        if reasoning_llm is None:
            raise RuntimeError("QueryUnderstandingService requires runtime.reasoning_llm. No reasoning model was provided.")
        parsed = invoke_structured(
            reasoning_llm,
            self._build_prompt(query, user_context),
            _QueryUnderstandingSchema,
            runtime=runtime,
            stage="query_understanding",
            role="reasoning",
            examples=[self._example_payload(query)],
            max_attempts=2,
        )
        return self._build_result(parsed, query)

    async def ainvoke(self, query: str, user_context: dict[str, Any], runtime: RuntimeContext) -> QueryUnderstandingResult:
        reasoning_llm = runtime.reasoning_llm
        if reasoning_llm is None:
            raise RuntimeError("QueryUnderstandingService requires runtime.reasoning_llm. No reasoning model was provided.")
        parsed = await ainvoke_structured(
            reasoning_llm,
            self._build_prompt(query, user_context),
            _QueryUnderstandingSchema,
            runtime=runtime,
            stage="query_understanding",
            role="reasoning",
            examples=[self._example_payload(query)],
            max_attempts=2,
        )
        return self._build_result(parsed, query)

    def _build_prompt(self, query: str, user_context: dict[str, Any]) -> str:
        context_lines = "\n".join(f"- {key}: {value}" for key, value in sorted(user_context.items())) or "- none"
        return (
            "You analyze requests for an NL2VIS system.\n"
            "Use the exact field names required by the schema.\n"
            "Do not classify requests into canonical/non-canonical groups.\n"
            "Generate query variants for these kinds when possible: canonical, schema_grounding, spec_retrieval, analysis.\n"
            f"User request:\n{query}\n\n"
            f"User context:\n{context_lines}\n"
        )

    def _build_result(self, parsed: _QueryUnderstandingSchema, original_query: str) -> QueryUnderstandingResult:
        return QueryUnderstandingResult(
            intent=parsed.intent.strip(),
            requested_operations=self._dedupe(parsed.requested_operations),
            candidate_charts=self._normalize_chart_names(parsed.candidate_charts),
            constraints=self._dedupe(parsed.constraints),
            case_type=None,
            confidence=parsed.confidence,
            task_type=parsed.task_type.strip(),
            user_goal=parsed.user_goal.strip(),
            analysis_goal=parsed.analysis_goal.strip(),
            query_variants=self._normalize_variants(parsed.query_variants, original_query),
            ambiguity_notes=self._dedupe(parsed.ambiguity_notes),
        )

    def _normalize_variants(self, values: list[_QueryVariantSchema], original_query: str) -> list[QueryVariant]:
        required = {"canonical", "schema_grounding", "spec_retrieval", "analysis"}
        seen: set[tuple[str, str]] = set()
        result: list[QueryVariant] = []
        for item in values:
            kind = item.kind.strip().lower().replace("-", "_")
            text = item.text.strip()
            key = (kind, text.lower())
            if kind and text and key not in seen:
                seen.add(key)
                result.append(QueryVariant(kind=kind, text=text, confidence=item.confidence, source="llm"))
        existing = {item.kind for item in result}
        for kind in sorted(required - existing):
            result.append(QueryVariant(kind=kind, text=original_query.strip(), confidence=0.51, source="derived"))
        return result

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
    def _normalize_chart_names(values: list[str]) -> list[str]:
        mapping = {
            "bar chart": "bar",
            "bar plot": "bar",
            "line chart": "line",
            "line plot": "line",
            "scatter plot": "scatter",
            "scatter chart": "scatter",
            "pie chart": "pie",
            "donut chart": "donut",
            "treemap": "treemap",
            "histogram": "histogram",
            "box plot": "boxplot",
            "box chart": "boxplot",
            "area chart": "area",
        }
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = value.strip().lower()
            mapped = mapping.get(normalized, normalized)
            if mapped and mapped not in seen:
                seen.add(mapped)
                result.append(mapped)
        return result

    @staticmethod
    def _example_payload(user_query: str) -> dict[str, Any]:
        return {
            "intent": f"Understand the analytical goal behind: {user_query}",
            "requested_operations": ["identify relevant grouping or trend operation"],
            "candidate_charts": ["bar", "line"],
            "constraints": [],
            "task_type": "descriptive_analytics",
            "user_goal": "understand the requested data slice visually",
            "analysis_goal": "extract the main visual takeaway",
            "confidence": 0.75,
            "query_variants": [
                {"kind": "canonical", "text": user_query, "confidence": 0.8},
                {"kind": "schema_grounding", "text": user_query, "confidence": 0.7},
            ],
            "ambiguity_notes": [],
        }
