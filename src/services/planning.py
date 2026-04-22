from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import (
    AnalysisRubric,
    CandidateSpecSet,
    DataProfile,
    ExecutionPolicy,
    PlanningResult,
    PlanningStep,
    QueryUnderstandingResult,
    RequestAnalysisResult,
    ValidationPolicy,
    VisRAGResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import ainvoke_structured, invoke_structured
from src.services.base import BaseService


class _ExecutionPolicySchema(BaseModel):
    max_retries: int = Field(default=1, ge=0, le=5)
    fallback_enabled: bool = True
    retry_strategy: str = Field(default="repair_then_fallback")
    prefer_best_ranked_spec: bool = True


class _ValidationPolicySchema(BaseModel):
    use_spec_validator: bool = True
    use_scenegraph_check: bool = True
    use_empty_chart_check: bool = True
    fail_fast_on_schema_error: bool = False


class _AnalysisRubricSchema(BaseModel):
    focus_areas: list[str] = Field(default_factory=list)
    output_format: str = Field(default="bullet_points")
    strict_visual_only: bool = True
    emphasize_anomalies: bool = True


class _PlanningSchema(BaseModel):
    steps: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    execution_policy: _ExecutionPolicySchema
    validation_policy: _ValidationPolicySchema
    analysis_rubric: _AnalysisRubricSchema


class PlanningService(BaseService):
    def invoke(self, query_understanding: QueryUnderstandingResult, request_analysis: RequestAnalysisResult, data_profile: DataProfile, visrag: VisRAGResult, runtime: RuntimeContext) -> PlanningResult:
        reasoning_llm = runtime.reasoning_llm
        if reasoning_llm is None:
            raise RuntimeError("PlanningService requires runtime.reasoning_llm. No reasoning model was provided.")
        candidate_spec_set = visrag.candidate_spec_set or CandidateSpecSet(
            candidate_specs=[],
            retrieved_examples=visrag.retrieved_examples,
            visualization_plan=visrag.visualization_plan,
            selected_candidate_spec=None,
        )
        parsed = invoke_structured(
            reasoning_llm,
            self._prompt(query_understanding, request_analysis, data_profile, visrag, candidate_spec_set),
            _PlanningSchema,
            runtime=runtime,
            stage="planning",
            role="reasoning",
            examples=[self._example_payload()],
            max_attempts=2,
        )
        return self._build_result(parsed)

    async def ainvoke(self, query_understanding: QueryUnderstandingResult, request_analysis: RequestAnalysisResult, data_profile: DataProfile, visrag: VisRAGResult, runtime: RuntimeContext) -> PlanningResult:
        reasoning_llm = runtime.reasoning_llm
        if reasoning_llm is None:
            raise RuntimeError("PlanningService requires runtime.reasoning_llm. No reasoning model was provided.")
        candidate_spec_set = visrag.candidate_spec_set or CandidateSpecSet(
            candidate_specs=[],
            retrieved_examples=visrag.retrieved_examples,
            visualization_plan=visrag.visualization_plan,
            selected_candidate_spec=None,
        )
        parsed = await ainvoke_structured(
            reasoning_llm,
            self._prompt(query_understanding, request_analysis, data_profile, visrag, candidate_spec_set),
            _PlanningSchema,
            runtime=runtime,
            stage="planning",
            role="reasoning",
            examples=[self._example_payload()],
            max_attempts=2,
        )
        return self._build_result(parsed)

    def _prompt(self, query_understanding: QueryUnderstandingResult, request_analysis: RequestAnalysisResult, data_profile: DataProfile, visrag: VisRAGResult, candidate_spec_set: CandidateSpecSet) -> str:
        return (
            "You create execution and analysis policies for an NL2VIS pipeline.\n"
            "Do not choose a new chart type. Use the supplied candidate specs and visualization plan.\n"
            f"Intent: {query_understanding.intent}\n"
            f"Task type: {query_understanding.task_type or 'unknown'}\n"
            f"User goal: {query_understanding.user_goal or 'unknown'}\n"
            f"Analysis goal: {query_understanding.analysis_goal or 'unknown'}\n"
            f"Ambiguity report: {', '.join(request_analysis.ambiguity_report)}\n"
            f"Missing fields: {', '.join(request_analysis.missing_fields)}\n"
            f"Data complexity: {data_profile.data_complexity or 'unknown'}\n"
            f"Candidate spec count: {len(candidate_spec_set.candidate_specs)}\n"
            f"Visualization plan JSON:\n{visrag.visualization_plan.model_dump_json(indent=2) if visrag.visualization_plan else '{}'}\n"
        )

    def _build_result(self, parsed: _PlanningSchema) -> PlanningResult:
        return PlanningResult(
            mode=None,
            steps=[PlanningStep(name=self._slugify(step), description=step) for step in self._dedupe(parsed.steps)],
            success_criteria=self._dedupe(parsed.success_criteria),
            execution_policy=ExecutionPolicy(**parsed.execution_policy.model_dump()),
            validation_policy=ValidationPolicy(**parsed.validation_policy.model_dump()),
            analysis_rubric=AnalysisRubric(**parsed.analysis_rubric.model_dump()),
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
    def _slugify(text: str) -> str:
        slug = "_".join(part for part in text.lower().replace("/", " ").replace("-", " ").split() if part)
        return slug[:60] or "step"

    @staticmethod
    def _example_payload() -> dict[str, object]:
        return {
            "steps": [
                "Use the top-ranked candidate specification.",
                "Validate the specification before rendering.",
            ],
            "success_criteria": [
                "The chart renders successfully.",
                "The chart remains analyzable in image-only mode.",
            ],
            "execution_policy": {
                "max_retries": 1,
                "fallback_enabled": True,
                "retry_strategy": "repair_then_fallback",
                "prefer_best_ranked_spec": True,
            },
            "validation_policy": {
                "use_spec_validator": True,
                "use_scenegraph_check": True,
                "use_empty_chart_check": True,
                "fail_fast_on_schema_error": False,
            },
            "analysis_rubric": {
                "focus_areas": ["trend", "peaks"],
                "output_format": "bullet_points",
                "strict_visual_only": True,
                "emphasize_anomalies": True,
            },
        }
