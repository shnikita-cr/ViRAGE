from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.domain.models import QueryUnderstandingResult, QueryVariant
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured
from src.services.base import BaseService


class _QueryUnderstandingSchema(BaseModel):
    intent: str = Field(min_length=1)
    requested_operations: list[str] = Field(default_factory=list)
    candidate_charts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    task_type: str = Field(min_length=1)
    user_goal: str = Field(min_length=1)
    analysis_goal: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    query_variants: list[QueryVariant] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)


class QueryUnderstandingService(BaseService):
    def invoke(self, query: str, user_context: dict[str, Any], runtime: RuntimeContext) -> QueryUnderstandingResult:
        reasoning_llm = runtime.reasoning_llm
        if reasoning_llm is None:
            raise RuntimeError(
                "QueryUnderstandingService requires runtime.reasoning_llm. No reasoning model was provided.")

        context_lines = "\n".join(f"- {key}: {value}" for key, value in sorted(user_context.items())) or "- none"
        prompt = (
            "You analyze user requests for an NL2VIS pipeline.\n"
            "Return only structured output.\n"
            "Do not classify the request into canonical or non-canonical groups.\n"
            "Infer the user's analytic intent, requested operations, likely chart families, constraints, "
            "task type, user goal, analysis goal, ambiguity notes, and a few useful query variants.\n"
            f"User request:\n{query}\n"
            f"User context:\n{context_lines}\n"
        )
        parsed = invoke_structured(reasoning_llm, prompt, _QueryUnderstandingSchema)
        return QueryUnderstandingResult(
            intent=parsed.intent.strip(),
            requested_operations=self._dedupe(parsed.requested_operations),
            candidate_charts=self._dedupe(parsed.candidate_charts),
            constraints=self._dedupe(parsed.constraints),
            case_type=None,
            confidence=parsed.confidence,
            task_type=parsed.task_type.strip(),
            user_goal=parsed.user_goal.strip(),
            analysis_goal=parsed.analysis_goal.strip(),
            query_variants=self._dedupe_variants(parsed.query_variants),
            ambiguity_notes=self._dedupe(parsed.ambiguity_notes),
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
    def _dedupe_variants(values: list[QueryVariant]) -> list[QueryVariant]:
        seen: set[tuple[str, str]] = set()
        result: list[QueryVariant] = []
        for item in values:
            key = (item.kind.strip().lower(), item.text.strip().lower())
            if key[0] and key[1] and key not in seen:
                seen.add(key)
                result.append(item)
        return result
