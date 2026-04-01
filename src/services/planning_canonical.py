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


class CanonicalPlanningService(BaseService):
    def invoke(self, query_understanding: QueryUnderstandingResult, runtime: RuntimeContext) -> PlanningResult:
        if runtime.llm is not None and is_langchain_available():
            try:
                return self._invoke_llm(query_understanding=query_understanding, runtime=runtime)
            except Exception:
                pass
        return self._invoke_fallback(query_understanding=query_understanding)

    def _invoke_llm(self, *, query_understanding: QueryUnderstandingResult, runtime: RuntimeContext) -> PlanningResult:
        prompt = (
            "Create a short deterministic plan for a canonical visualization case.\n"
            "The plan must be compact, operational and ordered.\n"
            "Avoid speculative steps. Prefer standard statistical visuals and checks.\n"
            f"Intent: {query_understanding.intent}\n"
            f"Requested operations: {', '.join(query_understanding.requested_operations)}\n"
            f"Candidate charts: {', '.join(query_understanding.candidate_charts)}\n"
            f"Constraints: {', '.join(query_understanding.constraints) or 'none'}\n"
        )
        parsed = invoke_structured(runtime.llm, prompt, _PlanningSchema)
        return PlanningResult(
            mode=ChartCaseType.CANONICAL,
            steps=[PlanningStep(name=self._slugify(step), description=step) for step in parsed.steps or self._fallback_steps(query_understanding)],
            success_criteria=self._dedupe(parsed.success_criteria or self._fallback_criteria(query_understanding)),
        )

    def _invoke_fallback(self, *, query_understanding: QueryUnderstandingResult) -> PlanningResult:
        steps = self._fallback_steps(query_understanding)
        return PlanningResult(
            mode=ChartCaseType.CANONICAL,
            steps=[PlanningStep(name=self._slugify(step), description=step) for step in steps],
            success_criteria=self._fallback_criteria(query_understanding),
        )

    def _fallback_steps(self, query_understanding: QueryUnderstandingResult) -> list[str]:
        ops = query_understanding.requested_operations
        charts = query_understanding.candidate_charts
        steps = [
            "Profile the dataset and confirm field types relevant to the request.",
            "Prepare a cleaned analysis-ready version of the data.",
            "Retrieve concise charting guidance for the selected chart family.",
        ]
        if "distribution analysis" in ops:
            steps.append("Build distribution-oriented visuals for the main numeric fields.")
        if "relationship analysis" in ops:
            steps.append("Build relationship-oriented visuals for the main numeric field pairs.")
        if "comparison" in ops or "ranking" in ops:
            steps.append("Build comparison-oriented visuals for grouped categories.")
        if not any(op in ops for op in ["distribution analysis", "relationship analysis", "comparison", "ranking"]):
            steps.append(f"Build the primary requested chart using the leading chart family: {charts[0]}.")
        steps.extend([
            "Execute plotting code and collect numeric summaries from the run.",
            "Read chart structure, extract facts and verify that final statements are evidence-backed.",
        ])
        return self._dedupe(steps)

    def _fallback_criteria(self, query_understanding: QueryUnderstandingResult) -> list[str]:
        charts = ", ".join(query_understanding.candidate_charts[:2]) or "the selected chart family"
        return [
            f"At least one valid canonical chart is produced, preferably among: {charts}.",
            "Generated charts are readable and consistent with the request.",
            "Final statements reference execution metrics or chart evidence.",
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
