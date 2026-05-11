from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.domain.models import StructuralSpecMetric, VisualQualityMetric


class BenchmarkCase(BaseModel):
    """Single VegaChat-compatible NL2VIS benchmark case."""

    case_id: str
    query: str
    data_path: str
    reference_spec: dict[str, Any] = Field(default_factory=dict)
    reference_image_path: str | None = None
    dataset_name: str = "unknown"
    difficulty: str | None = None
    utterance_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id", "query", "data_path")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("must be a non-empty string")
        return cleaned

    def resolved_data_path(self, root: Path) -> str:
        path = Path(self.data_path)
        return path.as_posix() if path.is_absolute() else (root / path).resolve().as_posix()

    def resolved_reference_image_path(self, root: Path) -> str | None:
        if not self.reference_image_path:
            return None
        path = Path(self.reference_image_path)
        return path.as_posix() if path.is_absolute() else (root / path).resolve().as_posix()


class BenchmarkCaseResult(BaseModel):
    case_id: str
    dataset_name: str = "unknown"
    query: str
    data_path: str
    run_id: str | None = None

    generated_spec: dict[str, Any] = Field(default_factory=dict)
    generated_image_path: str | None = None
    reference_image_path: str | None = None

    is_valid_spec: bool = False
    is_empty_chart: bool = False
    visualization_error_rate_item: bool = False
    empty_chart_rate_item: bool = False

    spec_score: float | None = None
    vision_score: float | None = None
    spec_metric: StructuralSpecMetric | None = None
    vision_metric: VisualQualityMetric | None = None
    metrics: dict[str, float] = Field(default_factory=dict)

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
            "query": self.query,
            "data_path": self.data_path,
            "run_id": self.run_id,
            "is_valid_spec": self.is_valid_spec,
            "is_empty_chart": self.is_empty_chart,
            "visualization_error_rate_item": self.visualization_error_rate_item,
            "empty_chart_rate_item": self.empty_chart_rate_item,
            "spec_score": self.spec_score,
            "vision_score": self.vision_score,
            "duration_seconds": self.duration_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "generated_image_path": self.generated_image_path,
            "reference_image_path": self.reference_image_path,
            "error": self.error,
            **{f"metric.{key}": value for key, value in self.metrics.items()},
            **{f"metadata.{key}": value for key, value in self.metadata.items()},
        }


class BenchmarkAggregateReport(BaseModel):
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    visualization_error_rate: float | None = None
    empty_chart_rate: float | None = None
    mean_spec_score: float | None = None
    mean_vision_score: float | None = None
    median_spec_score: float | None = None
    median_vision_score: float | None = None
    mean_duration_seconds: float | None = None
    total_tokens: int = 0
    vegachat_metrics: dict[str, float] = Field(default_factory=dict)
    results: list[BenchmarkCaseResult] = Field(default_factory=list)

    @classmethod
    def from_results(cls, results: list[BenchmarkCaseResult]) -> "BenchmarkAggregateReport":
        total = len(results)
        successful = sum(1 for item in results if item.error is None)
        failed = total - successful
        spec_scores = [float(item.spec_score) for item in results if item.spec_score is not None]
        vision_scores = [float(item.vision_score) for item in results if item.vision_score is not None]
        durations = [float(item.duration_seconds) for item in results if item.duration_seconds is not None]
        vegachat_metrics = _mean_metrics([item.metrics for item in results])
        return cls(
            total_cases=total,
            successful_cases=successful,
            failed_cases=failed,
            visualization_error_rate=_mean_bool([item.visualization_error_rate_item for item in results]),
            empty_chart_rate=_mean_bool([item.empty_chart_rate_item for item in results]),
            mean_spec_score=_mean(spec_scores),
            mean_vision_score=_mean(vision_scores),
            median_spec_score=_median(spec_scores),
            median_vision_score=_median(vision_scores),
            mean_duration_seconds=_mean(durations),
            total_tokens=sum(item.total_tokens for item in results),
            vegachat_metrics=vegachat_metrics,
            results=results,
        )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 6)
    return round((ordered[middle - 1] + ordered[middle]) / 2.0, 6)


def _mean_bool(values: list[bool]) -> float | None:
    return round(sum(1 for value in values if value) / len(values), 6) if values else None


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in metrics for key in item.keys()})
    out: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in metrics if key in item and item[key] is not None]
        values = [value for value in values if value == value]
        if values:
            out[key] = round(sum(values) / len(values), 6)
    return out
