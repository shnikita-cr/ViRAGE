from __future__ import annotations

from pathlib import Path

from src.benchmark.core.paths import resolve_path_from_root
from typing import Any

from src.benchmark.core.statistics import mean as _mean, mean_bool as _mean_bool, median as _median, percentile as _percentile

from pydantic import BaseModel, Field, field_validator, model_validator

from src.domain.models import StructuralSpecMetric, VisualQualityMetric


class BenchmarkCase(BaseModel):
    """Single VegaChat-compatible NL2VIS benchmark case."""

    case_id: str
    query: str
    data_path: str
    reference_spec: dict[str, Any] = Field(default_factory=dict)
    reference_specs: list[dict[str, Any]] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def _sync_reference_specs(self) -> "BenchmarkCase":
        if not self.reference_specs and self.reference_spec:
            self.reference_specs = [self.reference_spec]
        if self.reference_specs and not self.reference_spec:
            self.reference_spec = self.reference_specs[0]
        return self

    def effective_reference_specs(self) -> list[dict[str, Any]]:
        if self.reference_specs:
            return self.reference_specs
        return [self.reference_spec] if self.reference_spec else []

    def resolved_data_path(self, root: Path) -> str:
        return resolve_path_from_root(self.data_path, root)

    def resolved_reference_image_path(self, root: Path) -> str | None:
        if not self.reference_image_path:
            return None
        return resolve_path_from_root(self.reference_image_path, root)


class BenchmarkCaseResult(BaseModel):
    case_id: str
    dataset_name: str = "unknown"
    query: str
    data_path: str
    run_id: str | None = None

    generated_spec: dict[str, Any] = Field(default_factory=dict)
    generated_image_path: str | None = None
    reference_image_path: str | None = None
    reference_count: int | None = None
    best_reference_index: int | None = None
    reference_selection_method: str | None = None
    reference_render_error_count: int = 0
    reference_render_error_rate: float | None = None

    is_valid_spec: bool = False
    is_empty_chart: bool = False
    visualization_error_rate_item: bool = False
    empty_chart_rate_item: bool = False

    spec_score: float | None = None
    vision_score: float | None = None
    vlm_judge_score: float | None = None
    embedding_score: float | None = None
    clip_score: float | None = None
    siglip_score: float | None = None
    semantic_match_score: float | None = None
    technical_generation_attempts: int | None = None
    semantic_generation_attempts: int | None = None
    total_generation_attempts: int | None = None
    has_repeat_steps: bool | None = None
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
            "vlm_judge_score": self.vlm_judge_score,
            "embedding_score": self.embedding_score,
            "clip_score": self.clip_score,
            "siglip_score": self.siglip_score,
            "semantic_match_score": self.semantic_match_score,
            "technical_generation_attempts": self.technical_generation_attempts,
            "semantic_generation_attempts": self.semantic_generation_attempts,
            "total_generation_attempts": self.effective_total_generation_attempts(),
            "has_repeat_steps": self.effective_has_repeat_steps(),
            "duration_seconds": self.duration_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "generated_image_path": self.generated_image_path,
            "reference_image_path": self.reference_image_path,
            "reference_count": self.reference_count,
            "best_reference_index": self.best_reference_index,
            "reference_selection_method": self.reference_selection_method,
            "reference_render_error_count": self.reference_render_error_count,
            "reference_render_error_rate": self.reference_render_error_rate,
            "error": self.error,
            **{f"metric.{key}": value for key, value in self.metrics.items()},
            **{f"metadata.{key}": value for key, value in self.metadata.items()},
        }

    def effective_total_generation_attempts(self) -> int | None:
        if self.total_generation_attempts is not None:
            return self.total_generation_attempts
        if self.technical_generation_attempts is None and self.semantic_generation_attempts is None:
            return None
        return int(self.technical_generation_attempts or 0) + int(self.semantic_generation_attempts or 0)

    def effective_has_repeat_steps(self) -> bool | None:
        total_attempts = self.effective_total_generation_attempts()
        if total_attempts is None:
            return None
        return total_attempts > 1


class BenchmarkAggregateReport(BaseModel):
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    visualization_error_rate: float | None = None
    empty_chart_rate: float | None = None
    mean_spec_score: float | None = None
    mean_vision_score: float | None = None
    mean_spec_score_failure_as_zero: float | None = None
    mean_vision_score_failure_as_zero: float | None = None
    median_spec_score: float | None = None
    median_vision_score: float | None = None
    mean_vlm_judge_score: float | None = None
    median_vlm_judge_score: float | None = None
    mean_embedding_score: float | None = None
    median_embedding_score: float | None = None
    mean_semantic_match_score: float | None = None
    median_semantic_match_score: float | None = None
    mean_technical_generation_attempts_success: float | None = None
    mean_semantic_generation_attempts_success: float | None = None
    repeat_steps_rate: float | None = None
    repeat_steps_rate_success: float | None = None
    mean_attempts_success: float | None = None
    mean_technical_attempts_success: float | None = None
    mean_semantic_attempts_success: float | None = None
    mean_duration_seconds: float | None = None
    p95_duration_seconds: float | None = None
    mean_prompt_tokens: float | None = None
    mean_completion_tokens: float | None = None
    mean_total_tokens: float | None = None
    total_tokens: int = 0
    chart_text_consistency_rate: float | None = None
    mean_reference_count: float | None = None
    reference_render_error_rate: float | None = None
    best_reference_index_distribution: dict[str, int] = Field(default_factory=dict)
    stratified_metrics: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    vegachat_metrics: dict[str, float] = Field(default_factory=dict)
    sampling_strategy: str | None = None
    seed: int | None = None
    limit: int | None = None
    selected_case_ids: list[str] = Field(default_factory=list)
    selected_chart_types: list[str] = Field(default_factory=list)
    chart_type_distribution: dict[str, int] = Field(default_factory=dict)
    sampling_allocation: str | None = None
    cases_per_chart_type: int | None = None
    case_ids_hash: str | None = None
    sampling_warning: str | None = None
    results: list[BenchmarkCaseResult] = Field(default_factory=list)

    @classmethod
    def from_results(
            cls,
            results: list[BenchmarkCaseResult],
            *,
            sampling_metadata: dict[str, Any] | None = None,
    ) -> "BenchmarkAggregateReport":
        total = len(results)
        successful = sum(1 for item in results if item.error is None)
        failed = total - successful
        spec_scores = [float(item.spec_score) for item in results if item.spec_score is not None]
        vision_scores = [float(item.vision_score) for item in results if item.vision_score is not None]
        vlm_judge_scores = [float(item.vlm_judge_score) for item in results if item.vlm_judge_score is not None]
        embedding_scores = [float(item.embedding_score) for item in results if item.embedding_score is not None]
        semantic_match_scores = [float(item.semantic_match_score) for item in results if item.semantic_match_score is not None]
        technical_success_attempts = [float(item.technical_generation_attempts) for item in results if item.error is None and item.technical_generation_attempts is not None]
        semantic_success_attempts = [float(item.semantic_generation_attempts) for item in results if item.error is None and item.semantic_generation_attempts is not None]
        success_attempt_totals = [
            float(total_attempts)
            for item in results
            if item.error is None and (total_attempts := item.effective_total_generation_attempts()) is not None
        ]
        spec_scores_failure_as_zero = [
            float(item.spec_score) if item.spec_score is not None and item.error is None else 0.0 for item in results]
        vision_scores_failure_as_zero = [
            float(item.vision_score) if item.vision_score is not None and item.error is None else 0.0 for item in
            results]
        durations = [float(item.duration_seconds) for item in results if item.duration_seconds is not None]
        reference_counts = [float(item.reference_count) for item in results if item.reference_count is not None]
        reference_render_error_rates = [float(item.reference_render_error_rate) for item in results if item.reference_render_error_rate is not None]
        vegachat_metrics = _mean_metrics([item.metrics for item in results])
        text_consistency_values = [
            float(item.metrics["chart_text_consistency"])
            for item in results
            if "chart_text_consistency" in item.metrics
        ]
        sampling = sampling_metadata or {}
        return cls(
            total_cases=total,
            successful_cases=successful,
            failed_cases=failed,
            visualization_error_rate=_mean_bool([item.visualization_error_rate_item for item in results]),
            empty_chart_rate=_mean_bool([item.empty_chart_rate_item for item in results]),
            mean_spec_score=_mean(spec_scores),
            mean_vision_score=_mean(vision_scores),
            mean_spec_score_failure_as_zero=_mean(spec_scores_failure_as_zero),
            mean_vision_score_failure_as_zero=_mean(vision_scores_failure_as_zero),
            median_spec_score=_median(spec_scores),
            median_vision_score=_median(vision_scores),
            mean_vlm_judge_score=_mean(vlm_judge_scores),
            median_vlm_judge_score=_median(vlm_judge_scores),
            mean_embedding_score=_mean(embedding_scores),
            median_embedding_score=_median(embedding_scores),
            mean_semantic_match_score=_mean(semantic_match_scores),
            median_semantic_match_score=_median(semantic_match_scores),
            mean_technical_generation_attempts_success=_mean(technical_success_attempts),
            mean_semantic_generation_attempts_success=_mean(semantic_success_attempts),
            repeat_steps_rate=_repeat_steps_rate(results, successful_only=False),
            repeat_steps_rate_success=_repeat_steps_rate(results, successful_only=True),
            mean_attempts_success=_mean(success_attempt_totals),
            mean_technical_attempts_success=_mean(technical_success_attempts),
            mean_semantic_attempts_success=_mean(semantic_success_attempts),
            mean_duration_seconds=_mean(durations),
            p95_duration_seconds=_percentile(durations, 0.95),
            mean_prompt_tokens=_mean([float(item.prompt_tokens) for item in results]),
            mean_completion_tokens=_mean([float(item.completion_tokens) for item in results]),
            mean_total_tokens=_mean([float(item.total_tokens) for item in results]),
            total_tokens=sum(item.total_tokens for item in results),
            chart_text_consistency_rate=_mean(text_consistency_values),
            mean_reference_count=_mean(reference_counts),
            reference_render_error_rate=_mean(reference_render_error_rates),
            best_reference_index_distribution=_best_reference_index_distribution(results),
            stratified_metrics=_stratified_metrics(results),
            vegachat_metrics=vegachat_metrics,
            sampling_strategy=_maybe_str(sampling.get("sampling_strategy")),
            seed=_maybe_int(sampling.get("seed")),
            limit=_maybe_int(sampling.get("limit")),
            selected_case_ids=_maybe_str_list(sampling.get("selected_case_ids")),
            selected_chart_types=_maybe_str_list(sampling.get("selected_chart_types")),
            chart_type_distribution=_maybe_int_dict(sampling.get("chart_type_distribution")),
            sampling_allocation=_maybe_str(sampling.get("sampling_allocation")),
            cases_per_chart_type=_maybe_int(sampling.get("cases_per_chart_type")),
            case_ids_hash=_maybe_str(sampling.get("case_ids_hash")),
            sampling_warning=_maybe_str(sampling.get("sampling_warning")),
            results=results,
        )






def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]


def _maybe_int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        try:
            result[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return result


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in metrics for key in item.keys()})
    out: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in metrics if key in item and item[key] is not None]
        values = [value for value in values if value == value]
        if values:
            out[key] = round(sum(values) / len(values), 6)
    return out


def _stratified_metrics(results: list[BenchmarkCaseResult]) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[BenchmarkCaseResult]] = {}
    for item in results:
        for key in (
        "dataset_name", "utterance_type", "difficulty", "analysis_task", "chart_type", "reference_count"):
            value = item.dataset_name if key == "dataset_name" else item.metadata.get(key)
            if value is None or value == "":
                continue
            groups.setdefault(f"{key}:{value}", []).append(item)
    out: dict[str, dict[str, float | int | None]] = {}
    for group, items in sorted(groups.items()):
        successful_items = [item for item in items if item.error is None]
        technical_success_attempts = [
            float(item.technical_generation_attempts)
            for item in successful_items
            if item.technical_generation_attempts is not None
        ]
        semantic_success_attempts = [
            float(item.semantic_generation_attempts)
            for item in successful_items
            if item.semantic_generation_attempts is not None
        ]
        success_attempt_totals = [
            float(total_attempts)
            for item in successful_items
            if (total_attempts := item.effective_total_generation_attempts()) is not None
        ]
        out[group] = {
            "cases": len(items),
            "successful_cases": len(successful_items),
            "failed_cases": sum(1 for item in items if item.error is not None),
            "visualization_error_rate": _mean_bool([item.visualization_error_rate_item for item in items]),
            "empty_chart_rate": _mean_bool([item.empty_chart_rate_item for item in items]),
            "mean_spec_score": _mean([float(item.spec_score) for item in items if item.spec_score is not None]),
            "mean_vision_score": _mean([float(item.vision_score) for item in items if item.vision_score is not None]),
            "mean_spec_score_failure_as_zero": _mean([
                float(item.spec_score) if item.spec_score is not None and item.error is None else 0.0
                for item in items
            ]),
            "mean_vision_score_failure_as_zero": _mean([
                float(item.vision_score) if item.vision_score is not None and item.error is None else 0.0
                for item in items
            ]),
            "mean_total_tokens": _mean([float(item.total_tokens) for item in items]),
            "mean_duration_seconds": _mean(
                [float(item.duration_seconds) for item in items if item.duration_seconds is not None]),
            "repeat_steps_rate": _repeat_steps_rate(items, successful_only=False),
            "repeat_steps_rate_success": _repeat_steps_rate(items, successful_only=True),
            "mean_attempts_success": _mean(success_attempt_totals),
            "mean_technical_attempts_success": _mean(technical_success_attempts),
            "mean_semantic_attempts_success": _mean(semantic_success_attempts),
        }
    return out


def _best_reference_index_distribution(results: list[BenchmarkCaseResult]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for item in results:
        if item.best_reference_index is None:
            continue
        key = str(item.best_reference_index)
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _repeat_steps_rate(results: list[BenchmarkCaseResult], *, successful_only: bool) -> float | None:
    selected = [item for item in results if not successful_only or item.error is None]
    if not selected:
        return None
    return _mean_bool([bool(item.effective_has_repeat_steps()) for item in selected])
