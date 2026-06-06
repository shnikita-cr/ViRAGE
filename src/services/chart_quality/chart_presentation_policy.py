from __future__ import annotations

from typing import Any

import pandas as pd

from src.services.chart_quality._spec_utils import (
    LEGEND_CHANNELS,
    channel_field,
    clone_jsonish,
    field_values,
    get_encoding,
    is_discrete_channel,
    iter_unit_specs,
    legend_is_visible,
    set_axis_property,
)
from src.services.chart_quality.chart_quality_types import ChartPolicyResult, ChartQualityIssue


class ChartPresentationPolicy:
    def apply(self, spec: dict[str, Any], data: pd.DataFrame | None = None) -> ChartPolicyResult:
        clone = clone_jsonish(spec)
        issues: list[ChartQualityIssue] = []
        changes: list[str] = []
        for unit in iter_unit_specs(clone):
            encoding = get_encoding(unit)
            changes.extend(self._complete_titles_and_legends(unit, encoding, issues))
            changes.extend(self._fit_axis_labels(encoding, data, issues))
        changes.extend(self._fix_repeat_headers(clone, issues))
        return ChartPolicyResult(spec=clone, issues=issues, changes=changes)

    def _complete_titles_and_legends(
            self,
            unit: dict[str, Any],
            encoding: dict[str, Any],
            issues: list[ChartQualityIssue],
    ) -> list[str]:
        changes: list[str] = []
        title = unit.get("title")
        if not self._title_text(title):
            axis_fields = [channel_field(encoding.get(channel)) for channel in ("x", "y")]
            axis_fields = [field for field in axis_fields if field]
            if axis_fields:
                unit["title"] = " by ".join(axis_fields[:2])
                changes.append("set_missing_chart_title")
            else:
                issues.append(ChartQualityIssue(
                    code="missing_chart_title",
                    severity="warning",
                    message="chart title is missing",
                ))
        for channel in LEGEND_CHANNELS:
            channel_def = encoding.get(channel)
            if not isinstance(channel_def, dict):
                continue
            field = channel_field(channel_def)
            if not field:
                continue
            if channel_def.get("legend") is False or channel_def.get("legend") is None:
                if channel_def.get("legend") is False:
                    issues.append(ChartQualityIssue(
                        code="legend_disabled_for_encoded_field",
                        severity="warning",
                        message="encoded channel has no visible legend",
                        details={"channel": channel, "field": field},
                    ))
                    continue
                channel_def["legend"] = {"title": self._humanize(field)}
                changes.append(f"set_{channel}_legend_title")
            elif isinstance(channel_def.get("legend"), dict):
                channel_def["legend"].setdefault("title", self._humanize(field))
        return changes

    def _fit_axis_labels(
            self,
            encoding: dict[str, Any],
            data: pd.DataFrame | None,
            issues: list[ChartQualityIssue],
    ) -> list[str]:
        changes: list[str] = []
        x_def = encoding.get("x")
        y_def = encoding.get("y")
        if isinstance(x_def, dict) and is_discrete_channel(x_def):
            field = channel_field(x_def)
            longest, cardinality = self._label_stats(data, field)
            if longest >= 14 or cardinality >= 8:
                angle = self._x_label_angle(longest=longest, cardinality=cardinality)
                if angle is not None and set_axis_property(x_def, "labelAngle", angle, overwrite=False):
                    changes.append("set_x_label_angle_for_fit")
                if angle is not None:
                    set_axis_property(x_def, "labelAlign", "right", overwrite=False)
                    set_axis_property(x_def, "labelBaseline", "middle", overwrite=False)
                set_axis_property(x_def, "labelPadding", 8, overwrite=False)
                set_axis_property(x_def, "labelLimit", max(180, min(420, longest * 9)), overwrite=False)
                set_axis_property(x_def, "labelBound", True, overwrite=False)
            elif cardinality >= 4:
                set_axis_property(x_def, "labelOverlap", "greedy", overwrite=False)
                set_axis_property(x_def, "labelLimit", max(120, min(280, longest * 10)), overwrite=False)
            if cardinality >= 18 and longest >= 10:
                issues.append(ChartQualityIssue(
                    code="dense_x_category_labels",
                    severity="warning",
                    message="many long x-axis categories require filtering, ranking, or larger bottom margin",
                    details={"field": field, "cardinality": cardinality, "longest_label": longest},
                ))
        if isinstance(y_def, dict) and is_discrete_channel(y_def):
            field = channel_field(y_def)
            longest, cardinality = self._label_stats(data, field)
            set_axis_property(y_def, "labelLimit", max(180, min(460, longest * 10)), overwrite=False)
            set_axis_property(y_def, "labelBound", True, overwrite=False)
            if cardinality >= 16:
                issues.append(ChartQualityIssue(
                    code="dense_y_category_labels",
                    severity="warning",
                    message="many y-axis categories require sufficient height or top-N filtering",
                    details={"field": field, "cardinality": cardinality, "longest_label": longest},
                ))
        return changes

    @staticmethod
    def _x_label_angle(*, longest: int, cardinality: int) -> int | None:
        if cardinality <= 3 and longest <= 28:
            return None
        if cardinality <= 8 and longest <= 18:
            return -35
        return -90

    def _fix_repeat_headers(self, spec: dict[str, Any], issues: list[ChartQualityIssue]) -> list[str]:
        changes: list[str] = []
        repeat = spec.get("repeat")
        repeated_fields: list[str] = []
        if isinstance(repeat, dict):
            for key in ("column", "row"):
                values = repeat.get(key)
                if isinstance(values, list):
                    repeated_fields.extend(str(value) for value in values if value)
        elif isinstance(repeat, list):
            repeated_fields.extend(str(value) for value in repeat if value)
        if not repeated_fields:
            return changes
        header_title = self._repeat_title(repeated_fields)
        if not self._title_text(spec.get("title")):
            spec["title"] = header_title
            changes.append("set_repeat_chart_title")
        for unit in iter_unit_specs(spec):
            encoding = get_encoding(unit)
            for channel_name in ("x", "y"):
                channel_def = encoding.get(channel_name)
                if not isinstance(channel_def, dict):
                    continue
                field = channel_def.get("field")
                if isinstance(field, dict) and "repeat" in field:
                    axis = channel_def.setdefault("axis", {})
                    if isinstance(axis, dict):
                        current = str(axis.get("title") or "").strip().lower()
                        if current in {"", "value", "values", "metric", "measure"}:
                            axis["title"] = "Repeated metric value"
                            changes.append(f"set_repeat_{channel_name}_axis_title")
        if self._repeat_axis_title_is_generic(spec):
            issues.append(ChartQualityIssue(
                code="repeat_axis_title_generic",
                severity="warning",
                message="repeat chart axis title must explain that panel headers define the metric",
                details={"fields": repeated_fields},
            ))
        if not spec.get("resolve"):
            issues.append(ChartQualityIssue(
                code="repeat_scale_resolution_unspecified",
                severity="warning",
                message="repeat chart should state whether repeated fields share or use independent scales",
                details={"fields": repeated_fields},
            ))
        return changes

    @staticmethod
    def _repeat_title(repeated_fields: list[str]) -> str:
        labels = [ChartPresentationPolicy._humanize(field) for field in repeated_fields if str(field).strip()]
        if len(labels) <= 4:
            return f"Repeated metrics: {ChartPresentationPolicy._join_labels(labels)}"
        return f"Repeated metrics: {ChartPresentationPolicy._join_labels(labels[:3])} and {len(labels) - 3} more"

    @staticmethod
    def _join_labels(labels: list[str]) -> str:
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]
        return f"{', '.join(labels[:-1])} and {labels[-1]}"

    @staticmethod
    def _repeat_axis_title_is_generic(spec: dict[str, Any]) -> bool:
        generic = {"", "value", "values", "metric", "measure", "значение"}
        for unit in iter_unit_specs(spec):
            encoding = get_encoding(unit)
            for channel_name in ("x", "y"):
                channel_def = encoding.get(channel_name)
                if not isinstance(channel_def, dict):
                    continue
                field = channel_def.get("field")
                axis = channel_def.get("axis")
                if isinstance(field, dict) and "repeat" in field and isinstance(axis, dict):
                    if str(axis.get("title") or "").strip().lower() in generic:
                        return True
        return False

    @staticmethod
    def _label_stats(data: pd.DataFrame | None, field: str | None) -> tuple[int, int]:
        values = field_values(data, field)
        if not values:
            return len(str(field or "")), 1
        labels = [str(value) for value in values]
        return max(len(str(field or "")), max(len(label) for label in labels)), len(set(labels))

    @staticmethod
    def _title_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str):
                return text.strip()
            if isinstance(text, list):
                return " ".join(str(item).strip() for item in text if str(item).strip())
        return ""

    @staticmethod
    def _humanize(value: str) -> str:
        return value.replace("_", " ").strip().title()
