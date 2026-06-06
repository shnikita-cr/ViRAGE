from __future__ import annotations

from pathlib import Path

from src.benchmark.core.paths import resolve_path_from_root
from typing import Any

from src.benchmark.core.statistics import mean as _mean, mean_bool as _mean_bool, percentile as _percentile

from pydantic import BaseModel, Field, field_validator


class AnalysisBenchmarkCase(BaseModel):
    """Case format for chart-grounded analytical-agent evaluation.

    Compatible with simple InfiAgent-DABench/DABench exports after conversion to JSONL.
    """

    case_id: str
    question: str
    data_path: str
    expected_answer: str | None = None
    dataset_name: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id", "question", "data_path")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("must be a non-empty string")
        return text

    def resolved_data_path(self, root: Path) -> str:
        return resolve_path_from_root(self.data_path, root)


class AnalysisBenchmarkResult(BaseModel):
    case_id: str
    dataset_name: str = "unknown"
    question: str
    data_path: str
    expected_answer: str | None = None
    run_id: str | None = None
    generated_image_path: str | None = None
    chart_analysis_summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    chart_was_accepted: bool = False
    duration_seconds: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None
    evaluation_mode: str = "none"
    evaluation_verdict: str = "unknown"
    evaluation_score: float = 0.0
    chart_groundedness: float = 0.0
    hallucination_risk: float = 0.0
    exact_match: bool = False
    numeric_match_rate: float | None = None
    expected_items_count: int = 0
    matched_items_count: int = 0
    evaluation_rationale: str = ""
    artifact_run_dir: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_flat_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset_name": self.dataset_name,
            "question": self.question,
            "data_path": self.data_path,
            "expected_answer": self.expected_answer,
            "run_id": self.run_id,
            "generated_image_path": self.generated_image_path,
            "chart_analysis_summary": self.chart_analysis_summary,
            "key_findings": " | ".join(self.key_findings),
            "caveats": " | ".join(self.caveats),
            "confidence": self.confidence,
            "chart_was_accepted": self.chart_was_accepted,
            "duration_seconds": self.duration_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "evaluation_mode": self.evaluation_mode,
            "evaluation_verdict": self.evaluation_verdict,
            "evaluation_score": self.evaluation_score,
            "chart_groundedness": self.chart_groundedness,
            "hallucination_risk": self.hallucination_risk,
            "exact_match": self.exact_match,
            "numeric_match_rate": self.numeric_match_rate,
            "expected_items_count": self.expected_items_count,
            "matched_items_count": self.matched_items_count,
            "evaluation_rationale": self.evaluation_rationale,
            "artifact_run_dir": self.artifact_run_dir,
            "error": self.error,
            **{f"metadata.{key}": value for key, value in self.metadata.items()},
        }


class AnalysisBenchmarkReport(BaseModel):
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    accepted_chart_rate: float | None = None
    rejected_chart_rate: float | None = None
    technical_failure_rate: float | None = None
    mean_confidence: float | None = None
    mean_duration_seconds: float | None = None
    p95_duration_seconds: float | None = None
    mean_prompt_tokens: float | None = None
    mean_completion_tokens: float | None = None
    mean_total_tokens: float | None = None
    total_tokens: int = 0
    mean_evaluation_score: float | None = None
    mean_evaluation_score_failure_as_zero: float | None = None
    correct_rate: float | None = None
    partial_or_correct_rate: float | None = None
    mean_chart_groundedness: float | None = None
    mean_hallucination_risk: float | None = None
    stratified_metrics: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    results: list[AnalysisBenchmarkResult] = Field(default_factory=list)

    @classmethod
    def from_results(cls, results: list[AnalysisBenchmarkResult]) -> "AnalysisBenchmarkReport":
        total = len(results)
        success = sum(1 for item in results if item.error is None)
        technical_failures = sum(
            1 for item in results if item.error is not None and not str(item.error).startswith("chart_not_accepted"))
        accepted = [item.chart_was_accepted for item in results]
        rejected = [not item.chart_was_accepted and item.error is None for item in results]
        confidences = [item.confidence for item in results if item.error is None]
        durations = [item.duration_seconds for item in results if item.duration_seconds is not None]
        evaluated = [item for item in results if item.evaluation_verdict != "unknown"]
        evaluation_scores_failure_as_zero = [item.evaluation_score if item.error is None else 0.0 for item in results]
        return cls(
            total_cases=total,
            successful_cases=success,
            failed_cases=total - success,
            accepted_chart_rate=_mean_bool(accepted),
            rejected_chart_rate=_mean_bool(rejected),
            technical_failure_rate=round(technical_failures / total, 6) if total else None,
            mean_confidence=_mean(confidences),
            mean_duration_seconds=_mean(durations),
            p95_duration_seconds=_percentile(durations, 0.95),
            mean_prompt_tokens=_mean([float(item.prompt_tokens) for item in results]),
            mean_completion_tokens=_mean([float(item.completion_tokens) for item in results]),
            mean_total_tokens=_mean([float(item.total_tokens) for item in results]),
            total_tokens=sum(item.total_tokens for item in results),
            mean_evaluation_score=_mean([item.evaluation_score for item in evaluated]),
            mean_evaluation_score_failure_as_zero=_mean(evaluation_scores_failure_as_zero),
            correct_rate=_mean_bool([item.evaluation_verdict == "correct" for item in evaluated]),
            partial_or_correct_rate=_mean_bool(
                [item.evaluation_verdict in {"correct", "partially_correct"} for item in evaluated]),
            mean_chart_groundedness=_mean([item.chart_groundedness for item in evaluated]),
            mean_hallucination_risk=_mean([item.hallucination_risk for item in evaluated]),
            stratified_metrics=_stratified_metrics(results),
            results=results,
        )





def _stratified_metrics(results: list[AnalysisBenchmarkResult]) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[AnalysisBenchmarkResult]] = {}
    for item in results:
        for key in ("dataset_name", "level", "answerability_category"):
            value = item.dataset_name if key == "dataset_name" else item.metadata.get(key)
            if key == "answerability_category" and value is None:
                chart_answerability = item.metadata.get("chart_answerability")
                if isinstance(chart_answerability, dict):
                    value = chart_answerability.get("answerability_category") or chart_answerability.get("status")
            if value is None or value == "":
                continue
            groups.setdefault(f"{key}:{value}", []).append(item)
    out: dict[str, dict[str, float | int | None]] = {}
    for group, items in sorted(groups.items()):
        out[group] = {
            "cases": len(items),
            "technical_failures": sum(1 for item in items if item.error is not None),
            "accepted_chart_rate": _mean_bool([item.chart_was_accepted for item in items]),
            "mean_evaluation_score_failure_as_zero": _mean([
                item.evaluation_score if item.error is None else 0.0
                for item in items
            ]),
            "mean_chart_groundedness": _mean(
                [item.chart_groundedness for item in items if item.evaluation_verdict != "unknown"]),
            "mean_total_tokens": _mean([float(item.total_tokens) for item in items]),
            "mean_duration_seconds": _mean(
                [float(item.duration_seconds) for item in items if item.duration_seconds is not None]),
        }
    return out
