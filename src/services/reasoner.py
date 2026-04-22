from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import AnalysisRubric, InsightCandidate, InsightReasoningResult, VisualFactExtractionResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import ainvoke_structured, invoke_structured
from src.services.base import BaseService


class _ReasoningSchema(BaseModel):
    insight_candidates: list[InsightCandidate] = Field(default_factory=list)
    reasoning_chain: list[str] = Field(default_factory=list)


class ReasonerService(BaseService):
    def invoke(self, visual_facts: VisualFactExtractionResult, analysis_rubric: AnalysisRubric, runtime: RuntimeContext) -> InsightReasoningResult:
        if runtime.reasoning_llm is None:
            raise RuntimeError("Reasoning requires runtime.reasoning_llm. No reasoning model was provided.")
        prompt = (
            "You build insight candidates from visual-only facts extracted from a chart image.\n"
            f"Analysis rubric JSON:\n{analysis_rubric.model_dump_json(indent=2)}\n\n"
            f"Visual facts JSON:\n{visual_facts.model_dump_json(indent=2)}\n"
        )
        parsed = invoke_structured(
            runtime.reasoning_llm,
            prompt,
            _ReasoningSchema,
            runtime=runtime,
            stage="reasoner",
            role="reasoning",
            examples=[{
                "insight_candidates": [{
                    "statement": "The line trends upward over time.",
                    "confidence": 0.8,
                    "reasoning_chain": ["The plotted values increase from left to right."],
                }],
                "reasoning_chain": ["The plotted values increase from left to right."],
            }],
            max_attempts=2,
        )
        return InsightReasoningResult(**parsed.model_dump())

    async def ainvoke(self, visual_facts: VisualFactExtractionResult, analysis_rubric: AnalysisRubric, runtime: RuntimeContext) -> InsightReasoningResult:
        if runtime.reasoning_llm is None:
            raise RuntimeError("Reasoning requires runtime.reasoning_llm. No reasoning model was provided.")
        prompt = (
            "You build insight candidates from visual-only facts extracted from a chart image.\n"
            f"Analysis rubric JSON:\n{analysis_rubric.model_dump_json(indent=2)}\n\n"
            f"Visual facts JSON:\n{visual_facts.model_dump_json(indent=2)}\n"
        )
        parsed = await ainvoke_structured(
            runtime.reasoning_llm,
            prompt,
            _ReasoningSchema,
            runtime=runtime,
            stage="reasoner",
            role="reasoning",
            examples=[{
                "insight_candidates": [{
                    "statement": "The line trends upward over time.",
                    "confidence": 0.8,
                    "reasoning_chain": ["The plotted values increase from left to right."],
                }],
                "reasoning_chain": ["The plotted values increase from left to right."],
            }],
            max_attempts=2,
        )
        return InsightReasoningResult(**parsed.model_dump())
