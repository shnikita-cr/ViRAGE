from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.enums import ChartCaseType
from src.domain.models import PlanningResult, PlanningStep, QueryUnderstandingResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured, is_langchain_available
from src.services.base import BaseService


class _PlanningSchema(BaseModel):
    steps: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class NonCanonicalPlanningService(BaseService):
    def invoke(self, query_understanding: QueryUnderstandingResult, runtime: RuntimeContext) -> PlanningResult:
        if runtime.llm is not None and is_langchain_available():
            try:
                return self._invoke_llm(query_understanding=query_understanding, runtime=runtime)
            except Exception:
                pass
        return self._invoke_fallback(query_understanding=query_understanding)

    def _invoke_llm(self, *, query_understanding: QueryUnderstandingResult, runtime: RuntimeContext) -> PlanningResult:
        prompt = (
            "Create a cautious plan for a non-canonical visualization case.\n"
            "The plan must explicitly include assumption checks, fallback handling, and verification.\n"
            "Prefer conservative interpretation and stronger evidence collection.\n"
            f"Intent: {query_understanding.intent}\n"
            f"Requested operations: {', '.join(query_understanding.requested_operations)}\n"
            f"Candidate charts: {', '.join(query_understanding.candidate_charts)}\n"
            f"Constraints: {', '.join(query_understanding.constraints) or 'none'}\n"
        )
        parsed = invoke_structured(runtime.llm, prompt, _PlanningSchema)
        return PlanningResult(
            mode=ChartCaseType.NON_CANONICAL,
            steps=[PlanningStep(name=self._slugify(step), description=step) for step in parsed.steps or self._fallback_steps(query_understanding)],
            success_criteria=self._dedupe(parsed.success_criteria or self._fallback_criteria()),
        )

    def _invoke_fallback(self, *, query_understanding: QueryUnderstandingResult) -> PlanningResult:
        steps = self._fallback_steps(query_understanding)
        return PlanningResult(
            mode=ChartCaseType.NON_CANONICAL,
            steps=[PlanningStep(name=self._slugify(step), description=step) for step in steps],
            success_criteria=self._fallback_criteria(),
        )

    def _fallback_steps(self, query_understanding: QueryUnderstandingResult) -> list[str]:
        charts = ", ".join(query_understanding.candidate_charts[:3]) or "custom visuals"
        steps = [
            "Profile the dataset and identify fields that may support the requested non-canonical output.",
            "Prepare a cleaned and constrained working dataset to reduce ambiguity.",
            "Retrieve broader charting guidance, including caveats and anti-patterns, for the requested scenario.",
            f"Evaluate whether a non-canonical design is necessary or whether simpler chart families such as {charts} can satisfy the request.",
            "Generate and execute an initial chart solution together with numeric summaries.",
            "Read chart structure, extract evidence-backed facts and compare them against the intended message.",
            "Run an explicit verification pass to flag unsupported statements and unresolved assumptions.",
        ]
        return self._dedupe(steps)

    @staticmethod
    def _fallback_criteria() -> list[str]:
        return [
            "The produced chart addresses the user request without relying on unsupported visual assumptions.",
            "All final statements are evidence-backed or explicitly marked as uncertain.",
            "Verification identifies whether a simpler canonical alternative would have been sufficient.",
        ]

    @staticmethod
    def _slugify(text: str) -> str:
        return "_".join(text.lower().replace("-", " ").split()[:6])

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(value.strip())
        return result
