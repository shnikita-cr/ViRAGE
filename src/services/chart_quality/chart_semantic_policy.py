from __future__ import annotations

from typing import Any

import pandas as pd

from src.services.chart_quality._spec_utils import (
    channel_field,
    clone_jsonish,
    get_encoding,
    is_quantitative_channel,
    iter_unit_specs,
    mark_type,
    numeric_range,
    set_scale_property,
)
from src.services.chart_quality.chart_quality_types import ChartPolicyResult, ChartQualityIssue


class ChartSemanticPolicy:
    BAR_MARKS = {"bar"}
    COUNT_AGGREGATES = {"count", "valid", "missing", "sum"}
    LOCAL_DOMAIN_MARKS = {"boxplot", "point", "tick", "circle", "square", "line", "area", "rule"}

    def apply(self, spec: dict[str, Any], data: pd.DataFrame | None = None) -> ChartPolicyResult:
        clone = clone_jsonish(spec)
        issues: list[ChartQualityIssue] = []
        changes: list[str] = []
        for unit in iter_unit_specs(clone):
            encoding = get_encoding(unit)
            current_mark = mark_type(unit)
            for channel_name in ("x", "y"):
                channel_def = encoding.get(channel_name)
                if not isinstance(channel_def, dict) or not is_quantitative_channel(channel_def):
                    continue
                field = channel_field(channel_def)
                aggregate = str(channel_def.get("aggregate") or "").strip().lower()
                minimum, maximum = numeric_range(data, field)
                opposite_channel = "y" if channel_name == "x" else "x"
                opposite_def = encoding.get(opposite_channel)
                is_count_like = aggregate in self.COUNT_AGGREGATES
                if current_mark in self.BAR_MARKS or is_count_like:
                    if set_scale_property(channel_def, "zero", True, overwrite=True):
                        changes.append(f"set_{channel_name}_zero_true_for_{current_mark or 'aggregate'}")
                    if minimum is not None and maximum is not None and minimum > 0:
                        span = maximum - minimum
                        if maximum > 0 and span / max(abs(maximum), 1.0) < 0.12 and current_mark in self.BAR_MARKS:
                            issues.append(ChartQualityIssue(
                                code="bar_chart_may_hide_small_differences",
                                severity="warning",
                                message="bar chart keeps zero baseline; point or tick mark can show close values more accurately",
                                details={"field": field, "min": minimum, "max": maximum},
                            ))
                    continue
                if current_mark in self.LOCAL_DOMAIN_MARKS and not is_count_like:
                    if minimum is not None and maximum is not None:
                        span = maximum - minimum
                        if span > 0 and minimum > 0 and span / max(abs(maximum), 1.0) < 0.5:
                            if set_scale_property(channel_def, "zero", False, overwrite=False):
                                changes.append(f"set_{channel_name}_zero_false_for_local_comparison")
                if isinstance(opposite_def, dict) and opposite_def.get("axis") is None:
                    opposite_def["axis"] = {}
        issues.extend(self._shared_scale_metric_issues(clone, data))
        issues.extend(self._severity_usage_issues(clone, data))
        return ChartPolicyResult(spec=clone, issues=issues, changes=changes)

    def _fold_shared_scale_issues(self, spec: dict[str, Any], data: pd.DataFrame | None) -> list[ChartQualityIssue]:
        if data is None:
            return []
        issues: list[ChartQualityIssue] = []
        for unit in iter_unit_specs(spec):
            fold_fields = self._fold_fields(unit)
            if len(fold_fields) < 2:
                continue
            ranges: list[float] = []
            present_fields: list[str] = []
            for field in fold_fields:
                if field not in data.columns:
                    continue
                minimum, maximum = numeric_range(data, field)
                if minimum is None or maximum is None:
                    continue
                present_fields.append(field)
                ranges.append(max(abs(maximum - minimum), 1e-9))
            if len(ranges) >= 2 and max(ranges) / max(min(ranges), 1e-9) >= 8:
                issues.append(ChartQualityIssue(
                    code="folded_metrics_shared_scale",
                    severity="critical",
                    message="folded metrics have strongly different numeric ranges; use normalized severity or independent panels",
                    details={"fields": present_fields, "range_ratio": max(ranges) / max(min(ranges), 1e-9)},
                ))
        return issues

    @staticmethod
    def _fold_fields(unit: dict[str, Any]) -> list[str]:
        fields: list[str] = []
        transforms = unit.get("transform")
        if not isinstance(transforms, list):
            return fields
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            fold = transform.get("fold")
            if isinstance(fold, list):
                fields.extend(str(item) for item in fold if str(item).strip())
        return list(dict.fromkeys(fields))

    def _shared_scale_metric_issues(self, spec: dict[str, Any], data: pd.DataFrame | None) -> list[ChartQualityIssue]:
        issues: list[ChartQualityIssue] = []
        repeated = spec.get("repeat")
        fields: list[str] = []
        if isinstance(repeated, dict):
            for key in ("column", "row"):
                value = repeated.get(key)
                if isinstance(value, list):
                    fields.extend(str(item) for item in value if item)
        elif isinstance(repeated, list):
            fields.extend(str(item) for item in repeated if item)
        fields = [field for field in fields if data is not None and field in data.columns]
        if len(fields) >= 2:
            ranges: list[float] = []
            for field in fields:
                minimum, maximum = numeric_range(data, field)
                if minimum is None or maximum is None:
                    continue
                ranges.append(max(abs(maximum - minimum), 1e-9))
            if len(ranges) >= 2 and max(ranges) / max(min(ranges), 1e-9) >= 8:
                issues.append(ChartQualityIssue(
                    code="multi_metric_shared_scale_risk",
                    severity="critical",
                    message="repeated metrics have strongly different numeric ranges; use independent scales or normalized severity",
                    details={"fields": fields, "range_ratio": max(ranges) / max(min(ranges), 1e-9)},
                ))
                resolve = spec.setdefault("resolve", {})
                scale = resolve.setdefault("scale", {})
                scale.setdefault("y", "independent")
        issues.extend(self._fold_shared_scale_issues(spec, data))
        return issues

    def _severity_usage_issues(self, spec: dict[str, Any], data: pd.DataFrame | None) -> list[ChartQualityIssue]:
        if data is None or "overall_severity" not in data.columns:
            return []
        fields = self._encoded_fields(spec)
        if "overall_severity" in fields:
            return []
        return [ChartQualityIssue(
            code="severity_field_not_used",
            severity="critical",
            message="problematic-item data contains overall_severity, but the chart does not use it",
            details={"required_field": "overall_severity", "used_fields": sorted(fields)},
        )]

    @staticmethod
    def _encoded_fields(spec: dict[str, Any]) -> set[str]:
        fields: set[str] = set()
        for unit in iter_unit_specs(spec):
            encoding = get_encoding(unit)
            for channel_def in encoding.values():
                field = channel_field(channel_def)
                if field:
                    fields.add(field)
        return fields
