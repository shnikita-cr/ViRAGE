from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.domain.enums import ChartCaseType
from src.domain.models import QueryUnderstandingResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured, is_langchain_available
from src.services.base import BaseService


class _QueryUnderstandingSchema(BaseModel):
    intent: str = Field(min_length=1)
    requested_operations: list[str] = Field(default_factory=list)
    candidate_charts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    case_type: ChartCaseType
    confidence: float = Field(ge=0.0, le=1.0)


class QueryUnderstandingService(BaseService):
    def invoke(self, query: str, user_context: dict[str, Any], runtime: RuntimeContext) -> QueryUnderstandingResult:
        if runtime.llm is not None and is_langchain_available():
            try:
                return self._invoke_llm(query=query, user_context=user_context, runtime=runtime)
            except Exception:
                pass
        return self._invoke_fallback(query=query, user_context=user_context)

    def _invoke_llm(
            self,
            *,
            query: str,
            user_context: dict[str, Any],
            runtime: RuntimeContext,
    ) -> QueryUnderstandingResult:
        context_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(user_context.items())) or "- none"
        prompt = (
            "You analyze requests for a visualization pipeline.\n"
            "Return only structured output.\n"
            "Infer the user's analytic intent, requested operations, candidate chart families, constraints, "
            "case type and confidence.\n"
            "Use canonical when standard statistical or business charts are sufficient.\n"
            "Use non_canonical when the request implies unusual visual structure, infographic-like output, "
            "network/flow/diagram thinking, or ambiguous multimodal reasoning.\n"
            f"User request:\n{query}\n"
            f"User context:\n{context_lines}\n"
        )
        parsed = invoke_structured(runtime.llm, prompt, _QueryUnderstandingSchema)
        payload = parsed.model_dump()
        payload["requested_operations"] = self._dedupe(payload.get("requested_operations", []))
        payload["candidate_charts"] = self._dedupe(payload.get("candidate_charts", []))
        payload["constraints"] = self._dedupe(payload.get("constraints", []))
        if not payload["candidate_charts"]:
            payload["candidate_charts"] = ["bar", "line"]
        if not payload["requested_operations"]:
            payload["requested_operations"] = ["exploratory analysis"]
        return QueryUnderstandingResult(**payload)

    def _invoke_fallback(self, *, query: str, user_context: dict[str, Any]) -> QueryUnderstandingResult:
        q = query.lower().strip()
        charts: list[str] = []
        ops: list[str] = []
        constraints: list[str] = []

        keyword_map = {
            "trend analysis": ["trend", "over time", "timeline", "time series", "динам", "тренд"],
            "distribution analysis": ["distribution", "spread", "hist", "density", "распредел"],
            "comparison": ["compare", "comparison", "versus", "vs", "сравн"],
            "relationship analysis": ["correlation", "relationship", "scatter", "зависим", "связ"],
            "ranking": ["top", "bottom", "rank", "ranking", "лидер", "ранж"],
            "part-to-whole analysis": ["share", "composition", "part of whole", "доля", "состав"],
            "anomaly detection": ["anomaly", "outlier", "выброс", "аномал"],
        }
        chart_map = {
            "line": ["trend", "over time", "timeline", "time series", "динам", "тренд"],
            "scatter": ["correlation", "relationship", "scatter", "зависим", "связ"],
            "histogram": ["distribution", "hist", "density", "распредел"],
            "boxplot": ["spread", "outlier", "выброс", "распредел"],
            "bar": ["compare", "comparison", "top", "rank", "category", "сравн", "категор"],
            "heatmap": ["heatmap", "matrix", "correlation matrix", "матриц", "корреляц"],
            "area": ["share over time", "stacked area", "накоп"],
        }

        for op, hints in keyword_map.items():
            if any(h in q for h in hints):
                ops.append(op)
        for chart, hints in chart_map.items():
            if any(h in q for h in hints):
                charts.append(chart)

        if any(token in q for token in ["3d", "three dimensional", "объемн"]):
            constraints.append("avoid 3d unless explicitly required")
        if any(token in q for token in ["simple", "brief", "кратко", "прост"]):
            constraints.append("prefer concise visuals")
        if any(token in q for token in ["detailed", "подроб", "deep"]):
            constraints.append("allow richer analysis and multiple charts")
        if user_context.get("max_charts"):
            constraints.append(f"max_charts={user_context['max_charts']}")

        ambiguity_markers = [
            "interesting",
            "best",
            "better",
            "insightful",
            "look into",
            "something unusual",
            "интересн",
            "лучше",
            "что-нибудь",
        ]
        non_canonical_markers = [
            "diagram",
            "flow",
            "network",
            "mind map",
            "infographic",
            "story",
            "schema",
            "pipeline",
            "граф-схем",
            "схем",
            "инфограф",
            "сеть",
        ]
        case_type = ChartCaseType.CANONICAL
        if any(token in q for token in non_canonical_markers) or any(token in q for token in ambiguity_markers):
            case_type = ChartCaseType.NON_CANONICAL

        if not ops:
            ops = ["exploratory analysis"]
        if not charts:
            charts = ["line"] if "time" in q or "date" in q else ["bar", "line", "scatter"]

        intent = query.strip() or "exploratory analysis"
        confidence = 0.8 if case_type is ChartCaseType.CANONICAL else 0.55
        if any(token in q for token in ambiguity_markers):
            confidence -= 0.1
        return QueryUnderstandingResult(
            intent=intent,
            requested_operations=self._dedupe(ops),
            candidate_charts=self._dedupe(charts),
            constraints=self._dedupe(constraints),
            case_type=case_type,
            confidence=max(0.1, min(confidence, 0.99)),
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
