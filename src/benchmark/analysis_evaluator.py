from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured

EvaluationMode = Literal["none", "rules", "llm", "hybrid"]
_EXPECTED_TOKEN_RE = re.compile(r"@([A-Za-z0-9_\- .]+)\[([^\]]+)\]")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?(?:[eE][-+]?\d+)?")


class _LLMEvaluationSchema(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect", "unknown"] = "unknown"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    chart_groundedness: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class AnalysisEvaluationResult:
    mode: str
    verdict: str
    score: float
    chart_groundedness: float
    hallucination_risk: float
    exact_match: bool
    numeric_match_rate: float | None
    expected_items_count: int
    matched_items_count: int
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_mode": self.mode,
            "evaluation_verdict": self.verdict,
            "evaluation_score": round(float(self.score), 6),
            "chart_groundedness": round(float(self.chart_groundedness), 6),
            "hallucination_risk": round(float(self.hallucination_risk), 6),
            "exact_match": self.exact_match,
            "numeric_match_rate": self.numeric_match_rate,
            "expected_items_count": self.expected_items_count,
            "matched_items_count": self.matched_items_count,
            "evaluation_rationale": self.rationale,
        }


class HybridAnalysisEvaluator:
    def __init__(self, *, mode: EvaluationMode = "hybrid", numeric_tolerance: float = 0.03) -> None:
        self.mode = mode
        self.numeric_tolerance = max(0.0, numeric_tolerance)

    def evaluate(
        self,
        *,
        question: str,
        expected_answer: str | None,
        actual_answer: str,
        chart_summary: str,
        key_findings: list[str],
        caveats: list[str],
        runtime: RuntimeContext | None = None,
    ) -> AnalysisEvaluationResult:
        expected = str(expected_answer or "").strip()
        actual = str(actual_answer or "").strip()
        if self.mode == "none" or not expected:
            return AnalysisEvaluationResult(
                mode=self.mode,
                verdict="unknown",
                score=0.0,
                chart_groundedness=0.0,
                hallucination_risk=0.0,
                exact_match=False,
                numeric_match_rate=None,
                expected_items_count=0,
                matched_items_count=0,
                rationale="Evaluation skipped because mode is none or expected answer is empty.",
            )

        rule_result = self._evaluate_by_rules(expected=expected, actual=actual)
        if self.mode == "rules":
            return rule_result
        if self.mode == "hybrid" and rule_result.verdict == "correct":
            return rule_result
        if self.mode in {"llm", "hybrid"} and runtime is not None and runtime.reasoning_llm is not None:
            try:
                return self._evaluate_by_llm(
                    question=question,
                    expected_answer=expected,
                    actual_answer=actual,
                    chart_summary=chart_summary,
                    key_findings=key_findings,
                    caveats=caveats,
                    runtime=runtime,
                    fallback=rule_result,
                )
            except Exception as exc:
                return AnalysisEvaluationResult(
                    mode=f"{self.mode}_rules_fallback",
                    verdict=rule_result.verdict,
                    score=rule_result.score,
                    chart_groundedness=rule_result.chart_groundedness,
                    hallucination_risk=rule_result.hallucination_risk,
                    exact_match=rule_result.exact_match,
                    numeric_match_rate=rule_result.numeric_match_rate,
                    expected_items_count=rule_result.expected_items_count,
                    matched_items_count=rule_result.matched_items_count,
                    rationale=f"LLM evaluation failed: {type(exc).__name__}: {exc}. Rule fallback: {rule_result.rationale}",
                )
        return rule_result

    def _evaluate_by_rules(self, *, expected: str, actual: str) -> AnalysisEvaluationResult:
        expected_items = _extract_expected_items(expected)
        actual_normalized = _normalize_text(actual)
        expected_normalized = _normalize_text(expected)
        exact_match = bool(expected_normalized and expected_normalized in actual_normalized)

        matched = 0
        total = len(expected_items)
        if expected_items:
            actual_numbers = [_to_number(value) for value in _NUMBER_RE.findall(actual)]
            actual_numbers = [value for value in actual_numbers if value is not None]
            for key, value in expected_items:
                value_norm = _normalize_text(value)
                key_norm = _normalize_text(key)
                expected_number = _to_number(value)
                value_found = value_norm in actual_normalized if value_norm else False
                key_found = key_norm in actual_normalized if key_norm else True
                number_found = False
                if expected_number is not None:
                    number_found = any(_numbers_close(expected_number, item, tolerance=self.numeric_tolerance) for item in actual_numbers)
                if (value_found or number_found) and (key_found or total == 1):
                    matched += 1
            match_rate = matched / total if total else 0.0
        else:
            expected_numbers = [_to_number(value) for value in _NUMBER_RE.findall(expected)]
            expected_numbers = [value for value in expected_numbers if value is not None]
            actual_numbers = [_to_number(value) for value in _NUMBER_RE.findall(actual)]
            actual_numbers = [value for value in actual_numbers if value is not None]
            if expected_numbers:
                matched = sum(
                    1 for value in expected_numbers
                    if any(_numbers_close(value, item, tolerance=self.numeric_tolerance) for item in actual_numbers)
                )
                total = len(expected_numbers)
                match_rate = matched / total if total else 0.0
            else:
                total = 1
                matched = 1 if expected_normalized and expected_normalized in actual_normalized else 0
                match_rate = float(matched)

        if exact_match or match_rate >= 0.95:
            verdict = "correct"
            score = 1.0
        elif match_rate > 0.0:
            verdict = "partially_correct"
            score = round(match_rate, 6)
        else:
            verdict = "unknown"
            score = 0.0
        return AnalysisEvaluationResult(
            mode="rules",
            verdict=verdict,
            score=score,
            chart_groundedness=0.0,
            hallucination_risk=0.0,
            exact_match=exact_match,
            numeric_match_rate=round(match_rate, 6),
            expected_items_count=total,
            matched_items_count=matched,
            rationale=f"Rule evaluation matched {matched}/{total} expected items.",
        )

    def _evaluate_by_llm(
        self,
        *,
        question: str,
        expected_answer: str,
        actual_answer: str,
        chart_summary: str,
        key_findings: list[str],
        caveats: list[str],
        runtime: RuntimeContext,
        fallback: AnalysisEvaluationResult,
    ) -> AnalysisEvaluationResult:
        prompt = (
            "You are evaluating a chart-grounded data analysis benchmark result.\n"
            "Compare the actual answer with the expected answer. Do not reward unsupported claims.\n"
            "The system under evaluation was only allowed to analyze the accepted chart image for the final answer.\n"
            "Return whether the actual answer is correct, partially correct, incorrect, or unknown.\n\n"
            f"Question:\n{question}\n\n"
            f"Expected answer:\n{expected_answer}\n\n"
            f"Actual chart-grounded answer:\n{actual_answer}\n\n"
            f"Chart summary visible to evaluator:\n{chart_summary}\n\n"
            f"Key findings:\n{json.dumps(key_findings, ensure_ascii=False)}\n\n"
            f"Caveats:\n{json.dumps(caveats, ensure_ascii=False)}\n\n"
            "Scoring guidance: correct=1.0, partially_correct=0.3..0.8, incorrect=0.0, unknown=0.0. "
            "chart_groundedness is 0..1. hallucination_risk is 0..1."
        )
        parsed = invoke_structured(
            runtime.reasoning_llm,
            prompt,
            _LLMEvaluationSchema,
            runtime=runtime,
            stage="analysis_benchmark_evaluation",
            role="reasoning",
            examples=[{
                "verdict": "partially_correct",
                "score": 0.5,
                "chart_groundedness": 0.7,
                "hallucination_risk": 0.2,
                "rationale": "The answer contains the direction but misses one expected value.",
            }],
            max_attempts=2,
        )
        return AnalysisEvaluationResult(
            mode="llm" if self.mode == "llm" else "hybrid_llm",
            verdict=parsed.verdict,
            score=float(parsed.score),
            chart_groundedness=float(parsed.chart_groundedness),
            hallucination_risk=float(parsed.hallucination_risk),
            exact_match=fallback.exact_match,
            numeric_match_rate=fallback.numeric_match_rate,
            expected_items_count=fallback.expected_items_count,
            matched_items_count=fallback.matched_items_count,
            rationale=parsed.rationale,
        )


def _extract_expected_items(text: str) -> list[tuple[str, str]]:
    return [(match.group(1).strip(), match.group(2).strip()) for match in _EXPECTED_TOKEN_RE.finditer(text or "")]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower().replace(",", "."))


def _to_number(value: str) -> float | None:
    try:
        number = float(str(value).strip().replace(",", "."))
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _numbers_close(left: float, right: float, *, tolerance: float) -> bool:
    absolute_tolerance = max(tolerance, abs(left) * tolerance)
    return abs(left - right) <= absolute_tolerance
