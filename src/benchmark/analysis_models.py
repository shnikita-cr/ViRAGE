from __future__ import annotations

from pathlib import Path
from typing import Any

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
        path = Path(self.data_path)
        return path.as_posix() if path.is_absolute() else (root / path).resolve().as_posix()


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
            "error": self.error,
            **{f"metadata.{key}": value for key, value in self.metadata.items()},
        }


class AnalysisBenchmarkReport(BaseModel):
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    accepted_chart_rate: float | None = None
    mean_confidence: float | None = None
    mean_duration_seconds: float | None = None
    total_tokens: int = 0
    results: list[AnalysisBenchmarkResult] = Field(default_factory=list)

    @classmethod
    def from_results(cls, results: list[AnalysisBenchmarkResult]) -> "AnalysisBenchmarkReport":
        total = len(results)
        success = sum(1 for item in results if item.error is None)
        accepted = [item.chart_was_accepted for item in results if item.error is None]
        confidences = [item.confidence for item in results if item.error is None]
        durations = [item.duration_seconds for item in results if item.duration_seconds is not None]
        return cls(
            total_cases=total,
            successful_cases=success,
            failed_cases=total - success,
            accepted_chart_rate=_mean_bool(accepted),
            mean_confidence=_mean(confidences),
            mean_duration_seconds=_mean(durations),
            total_tokens=sum(item.total_tokens for item in results),
            results=results,
        )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _mean_bool(values: list[bool]) -> float | None:
    return round(sum(1 for value in values if value) / len(values), 6) if values else None
