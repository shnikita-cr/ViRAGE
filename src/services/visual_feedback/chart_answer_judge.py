from __future__ import annotations

import json

from pydantic import BaseModel, Field

from src.domain.models import ChartAnswerJudgeResult, ChartFactSummaryResult, RequestAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured
from src.services.base import BaseService


class _ChartAnswerJudgeSchema(BaseModel):
    answers_user_query: bool = False
    confidence: float = 0.0
    retry_recommendation: str = "retry"
    missing_requirements: list[str] = Field(default_factory=list)
    wrong_or_suspicious_parts: list[str] = Field(default_factory=list)
    improvement_comments: list[str] = Field(default_factory=list)
    feedback_for_next_generation: str = ""


class ChartAnswerJudgeService(BaseService):
    def invoke(
        self,
        *,
        query: str,
        chart_facts: ChartFactSummaryResult,
        runtime: RuntimeContext,
        request_analysis: RequestAnalysisResult | None = None,
    ) -> ChartAnswerJudgeResult:
        if runtime.reasoning_llm is None:
            raise RuntimeError("Chart answer judge requires runtime.reasoning_llm.")
        request_payload = {
            "query": query,
            "request_analysis": request_analysis.model_dump() if request_analysis is not None else None,
            "chart_facts": chart_facts.model_dump(),
        }
        prompt = (
            "You are ChartAnswerJudgeAI. Decide whether the visible chart facts answer the user request.\n"
            "Use only the user request, request analysis, and chart facts. Do not assume hidden data.\n"
            "If the chart is technically rendered but semantically insufficient, return retry_recommendation='retry' "
            "and write concrete feedback for the next Vega-Lite generation.\n\n"
            f"Payload JSON:\n{json.dumps(request_payload, ensure_ascii=False, indent=2)}\n"
        )
        parsed = invoke_structured(
            runtime.reasoning_llm,
            prompt,
            _ChartAnswerJudgeSchema,
            runtime=runtime,
            stage="chart_answer_judge",
            role="reasoner",
            examples=[{
                "answers_user_query": False,
                "confidence": 0.55,
                "retry_recommendation": "retry",
                "missing_requirements": ["The requested metric is not visible."],
                "wrong_or_suspicious_parts": [],
                "improvement_comments": ["Use the requested metric and group by the requested category."],
                "feedback_for_next_generation": "The previous chart did not show the requested metric. Generate a chart using the requested metric and grouping fields.",
            }],
            max_attempts=2,
        )
        recommendation = str(parsed.retry_recommendation).strip().lower()
        if recommendation not in {"accept", "retry", "reject"}:
            recommendation = "retry"
        return ChartAnswerJudgeResult(
            answers_user_query=bool(parsed.answers_user_query),
            confidence=max(0.0, min(1.0, float(parsed.confidence or 0.0))),
            retry_recommendation=recommendation,  # type: ignore[arg-type]
            missing_requirements=list(parsed.missing_requirements),
            wrong_or_suspicious_parts=list(parsed.wrong_or_suspicious_parts),
            improvement_comments=list(parsed.improvement_comments),
            feedback_for_next_generation=parsed.feedback_for_next_generation,
        )
