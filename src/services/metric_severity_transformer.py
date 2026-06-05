from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.orchestrator.planning_contract import (
    allowed_metric_semantics,
    is_problem_ranking_strategy,
    requires_severity_fields,
)


@dataclass(frozen=True)
class MetricSeverityTransformResult:
    frame: pd.DataFrame
    added_columns: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricSeverityTransformer:
    LOWER_IS_BETTER = {"lower_is_better", "higher_is_worse", "count_higher_is_worse", "percentage_higher_is_worse"}
    HIGHER_IS_BETTER = {"higher_is_better", "lower_is_worse"}
    DEVIATION_IS_BAD = {"deviation_from_typical_is_bad"}
    NEUTRAL = {"neutral_measurement"}

    def apply(
            self,
            frame: pd.DataFrame,
            *,
            metric_semantics: dict[str, Any] | None,
            ranking_strategy: str | None = None,
            selected_fields: list[str] | None = None,
            max_ranked_rows: int = 12,
    ) -> MetricSeverityTransformResult:
        semantics = self._normalize_semantics(metric_semantics or {})
        if not semantics:
            self._raise_if_required(ranking_strategy, "metric_semantics is empty")
            return MetricSeverityTransformResult(frame=frame)

        selected = {str(field) for field in selected_fields or [] if str(field).strip()}
        clone = frame.copy()
        added, used_metrics = self._add_metric_severity_columns(clone, semantics, selected)
        operations: list[str] = []

        if added:
            clone["overall_severity"] = clone[added].mean(axis=1, skipna=True)
            added.append("overall_severity")
            operations.append("derive_metric_severity")
        else:
            self._raise_if_required(ranking_strategy, "no severity columns were derived")

        if "overall_severity" in clone.columns and is_problem_ranking_strategy(ranking_strategy):
            clone, limit_operation = self._rank_problematic_rows(clone, max_ranked_rows=max_ranked_rows)
            operations.extend(limit_operation)

        return MetricSeverityTransformResult(
            frame=clone,
            added_columns=added,
            operations=operations,
            metadata={"used_metrics": used_metrics, "ranking_strategy": ranking_strategy, "max_ranked_rows": max_ranked_rows},
        )

    def _add_metric_severity_columns(
            self,
            frame: pd.DataFrame,
            semantics: dict[str, str],
            selected_fields: set[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        added: list[str] = []
        used_metrics: list[dict[str, Any]] = []
        for field, direction in semantics.items():
            if not self._field_is_eligible(frame, field, selected_fields):
                continue
            severity = self._severity(pd.to_numeric(frame[field], errors="coerce"), direction)
            if severity is None:
                continue
            severity_name = _unique_column_name(frame, f"severity_{field}")
            frame[severity_name] = severity
            added.append(severity_name)
            used_metrics.append({"field": field, "direction": direction, "severity_field": severity_name})
        return added, used_metrics

    @staticmethod
    def _field_is_eligible(frame: pd.DataFrame, field: str, selected_fields: set[str]) -> bool:
        if field not in frame.columns:
            return False
        if selected_fields and field not in selected_fields:
            return False
        return pd.to_numeric(frame[field], errors="coerce").notna().sum() >= 2

    def _severity(self, series: pd.Series, direction: str) -> pd.Series | None:
        if direction in self.NEUTRAL:
            return None
        scaled = _minmax(series)
        if scaled is None:
            return None
        if direction in self.LOWER_IS_BETTER:
            return scaled
        if direction in self.HIGHER_IS_BETTER:
            return 1.0 - scaled
        if direction in self.DEVIATION_IS_BAD:
            return _minmax((series - series.median(skipna=True)).abs())
        raise ValueError(f"Unsupported metric semantic: {direction!r}. Allowed: {allowed_metric_semantics()}")

    @staticmethod
    def _rank_problematic_rows(frame: pd.DataFrame, *, max_ranked_rows: int) -> tuple[pd.DataFrame, list[str]]:
        operations = ["sort_by_overall_severity_desc"]
        ranked = frame.sort_values("overall_severity", ascending=False, kind="mergesort")
        limit = max(1, int(max_ranked_rows))
        if len(ranked) > limit:
            return ranked.head(limit).copy(), [*operations, f"limit_problematic_top_n:{limit}"]
        return ranked.copy(), operations

    @staticmethod
    def _raise_if_required(ranking_strategy: str | None, reason: str) -> None:
        if requires_severity_fields(ranking_strategy):
            raise ValueError(f"Severity ranking requires derived severity fields: {reason}.")

    @staticmethod
    def _normalize_semantics(value: dict[str, Any]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, raw in value.items():
            field = str(key).strip()
            direction = _extract_direction(raw)
            if not field or direction == "neutral_measurement":
                continue
            if direction not in set(allowed_metric_semantics()):
                raise ValueError(f"Unsupported metric semantic for {field!r}: {direction!r}. Allowed: {allowed_metric_semantics()}")
            result[field] = direction
        return result


def _extract_direction(value: Any) -> str:
    if isinstance(value, dict):
        raw = value.get("direction") or value.get("quality_direction") or value.get("semantic") or value.get("role")
    else:
        raw = value
    return str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")


def _minmax(series: pd.Series) -> pd.Series | None:
    valid = series.dropna()
    if valid.empty:
        return None
    low = float(valid.min())
    high = float(valid.max())
    if high == low:
        return pd.Series(0.0, index=series.index, dtype="float64").where(series.notna())
    return ((series - low) / (high - low)).clip(0.0, 1.0)


def _unique_column_name(frame: pd.DataFrame, base: str) -> str:
    candidate = base
    suffix = 2
    while candidate in frame.columns:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate
