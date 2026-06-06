from __future__ import annotations

import json

from pydantic import BaseModel, Field

from src.domain.models import ChartAnswerJudgeResult, ChartFactSummaryResult, QueryRequestAnalysisResult
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
            request_analysis: QueryRequestAnalysisResult | None = None,
    ) -> ChartAnswerJudgeResult:
        if runtime.reasoning_llm is None:
            raise RuntimeError("Chart answer judge requires runtime.reasoning_llm.")
        request_payload = {
            "query": query,
            "request_analysis": request_analysis.model_dump() if request_analysis is not None else None,
            "chart_facts": chart_facts.model_dump(),
        }
        prompt = (
            "You are a chart-answer evaluator. Check whether the chart can answer the user question using only visible evidence. Decide whether the visible chart facts answer the user request.\n"
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
        missing_requirements = _clean_text_list(parsed.missing_requirements)
        wrong_or_suspicious_parts = _clean_text_list(parsed.wrong_or_suspicious_parts)
        improvement_comments = _clean_text_list(parsed.improvement_comments)
        feedback_for_next_generation = str(parsed.feedback_for_next_generation or "").strip()
        recommendation = _normalize_retry_recommendation(
            parsed.retry_recommendation,
            answers_user_query=bool(parsed.answers_user_query),
            feedback_for_next_generation=feedback_for_next_generation,
            missing_requirements=missing_requirements,
            wrong_or_suspicious_parts=wrong_or_suspicious_parts,
            improvement_comments=improvement_comments,
        )
        return ChartAnswerJudgeResult(
            answers_user_query=bool(parsed.answers_user_query),
            confidence=max(0.0, min(1.0, float(parsed.confidence or 0.0))),
            retry_recommendation=recommendation,
            missing_requirements=missing_requirements,
            wrong_or_suspicious_parts=wrong_or_suspicious_parts,
            improvement_comments=improvement_comments,
            feedback_for_next_generation=feedback_for_next_generation,
        )


def _clean_text_list(values: list[str]) -> list[str]:
    return [str(item).strip() for item in values if str(item).strip()]


def _normalize_retry_recommendation(
        value: object,
        *,
        answers_user_query: bool,
        feedback_for_next_generation: str,
        missing_requirements: list[str],
        wrong_or_suspicious_parts: list[str],
        improvement_comments: list[str],
) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    actionable_feedback_exists = bool(
        feedback_for_next_generation
        or missing_requirements
        or wrong_or_suspicious_parts
        or improvement_comments
    )

    if raw in {"accept", "accepted", "ok", "okay", "pass", "passed", "no_retry", "no_retries", "none", "not_retry",
               "do_not_retry"}:
        return "accept"
    if raw in {"retry", "revise", "regenerate", "needs_retry", "needs_improvement", "fix"}:
        return "retry"
    if raw in {"reject", "rejected", "fail", "failed"}:
        return "reject"

    # Be conservative only when the judge found concrete issues. Otherwise a positive answer should not
    # trigger a silent retry just because the model used a non-standard recommendation token.
    if answers_user_query and not actionable_feedback_exists:
        return "accept"
    return "retry" if actionable_feedback_exists else "reject"
