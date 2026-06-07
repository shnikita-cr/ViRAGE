from __future__ import annotations

from pathlib import Path
from typing import Any

from src.services.spec.repeat_labels import title_text

import pandas as pd

from src.services.chart_quality.common.spec_utils import get_encoding, iter_unit_specs, mark_type
from src.services.chart_quality.chart_quality_types import ChartQualityIssue, ChartQualityReport, ChartQualityThresholds


class ChartQualityEvaluator:
    def __init__(
            self,
            thresholds: ChartQualityThresholds | None = None,
            *,
            max_warnings_for_pass: int = 2,
    ) -> None:
        if max_warnings_for_pass < 0:
            raise ValueError("max_warnings_for_pass must be non-negative.")
        self.thresholds = thresholds or ChartQualityThresholds()
        self.hard_fail_codes = set(self.thresholds.hard_fail_codes)
        self.max_warnings_for_pass = max_warnings_for_pass

    def evaluate(
            self,
            *,
            spec: dict[str, Any],
            png_path: str | Path | None = None,
            scenegraph_summary: dict[str, Any] | None = None,
            data: pd.DataFrame | None = None,
            policy_issues: list[ChartQualityIssue] | None = None,
    ) -> ChartQualityReport:
        issues: list[ChartQualityIssue] = list(policy_issues or [])
        metrics = self._collect_metrics(spec=spec, png_path=png_path, scenegraph_summary=scenegraph_summary, data=data, issues=issues)
        issue_counts = self._issue_counts(issues)
        status = self._status(issues=issues, critical=issue_counts["critical"], warnings=issue_counts["warning"])
        score = max(0.0, 1.0 - issue_counts["critical"] * 0.4 - issue_counts["warning"] * 0.08)
        return ChartQualityReport(status=status, score=round(score, 4), issues=issues, metrics=metrics)

    def _collect_metrics(
            self,
            *,
            spec: dict[str, Any],
            png_path: str | Path | None,
            scenegraph_summary: dict[str, Any] | None,
            data: pd.DataFrame | None,
            issues: list[ChartQualityIssue],
    ) -> dict[str, Any]:
        metrics = self._size_metrics(spec, png_path, issues)
        metrics.update(self._spec_metrics(spec, scenegraph_summary, issues))
        metrics.update(self._scenegraph_metrics(scenegraph_summary, issues))
        issues.extend(self._dense_category_issues(spec, data))
        return metrics

    def _status(self, *, issues: list[ChartQualityIssue], critical: int, warnings: int) -> str:
        if any(issue.code in self.hard_fail_codes for issue in issues):
            return "fail"
        if critical:
            return "retry"
        if warnings > self.max_warnings_for_pass:
            return "retry"
        return "pass"

    @staticmethod
    def _issue_counts(issues: list[ChartQualityIssue]) -> dict[str, int]:
        return {
            "critical": sum(1 for issue in issues if issue.severity == "critical"),
            "warning": sum(1 for issue in issues if issue.severity == "warning"),
            "info": sum(1 for issue in issues if issue.severity == "info"),
        }

    def _size_metrics(self, spec: dict[str, Any], png_path: str | Path | None, issues: list[ChartQualityIssue]) -> dict[str, Any]:
        logical_width = self._integer_value(spec.get("width"))
        logical_height = self._integer_value(spec.get("height"))
        pixel_width, pixel_height = self._png_size(Path(png_path)) if png_path else (0, 0)
        if pixel_width and pixel_height:
            self._append_canvas_issues(pixel_width=pixel_width, pixel_height=pixel_height, issues=issues)
        return {"logical_width": logical_width, "logical_height": logical_height, "pixel_width": pixel_width, "pixel_height": pixel_height}

    @staticmethod
    def _integer_value(value: Any) -> int:
        return int(value) if isinstance(value, int) else 0

    def _append_canvas_issues(self, *, pixel_width: int, pixel_height: int, issues: list[ChartQualityIssue]) -> None:
        aspect = pixel_width / max(pixel_height, 1)
        if pixel_width > self.thresholds.max_rendered_width_px:
            issues.append(ChartQualityIssue("excessive_rendered_width", "warning", "rendered image exceeds configured width", {"pixel_width": pixel_width}))
        if pixel_height > self.thresholds.max_rendered_height_px:
            issues.append(ChartQualityIssue("excessive_rendered_height", "warning", "rendered image exceeds configured height", {"pixel_height": pixel_height}))
        if aspect > self.thresholds.max_aspect_ratio:
            issues.append(ChartQualityIssue("excessive_wide_canvas", "warning", "rendered image is very wide relative to height", {"aspect_ratio": round(aspect, 3)}))
        if aspect < self.thresholds.min_aspect_ratio:
            issues.append(ChartQualityIssue("excessive_tall_canvas", "warning", "rendered image is very tall relative to width", {"aspect_ratio": round(aspect, 3)}))

    def _spec_metrics(self, spec: dict[str, Any], scenegraph_summary: dict[str, Any] | None, issues: list[ChartQualityIssue]) -> dict[str, Any]:
        unit_specs = list(iter_unit_specs(spec))
        uses_color, visible_legend_required = self._inspect_unit_specs(unit_specs, issues)
        has_legend = bool((scenegraph_summary or {}).get("has_legend"))
        repeat_used = bool(spec.get("repeat"))
        if uses_color and visible_legend_required and not has_legend:
            issues.append(ChartQualityIssue("required_legend_not_rendered", "critical", "color encoding requires a visible legend"))
        if repeat_used and not self._repeat_labels_are_specific(spec):
            issues.append(ChartQualityIssue("repeat_labels_invalid", "critical", "repeat chart must expose concrete metric labels in title, axis or panel headers"))
        return {"unit_spec_count": len(unit_specs), "uses_color": uses_color, "has_legend": has_legend, "repeat_used": repeat_used}

    def _inspect_unit_specs(self, unit_specs: list[dict[str, Any]], issues: list[ChartQualityIssue]) -> tuple[bool, bool]:
        uses_color = False
        visible_legend_required = False
        for unit in unit_specs:
            encoding = get_encoding(unit)
            color = encoding.get("color")
            if isinstance(color, dict) and color.get("field"):
                uses_color = True
                visible_legend_required = visible_legend_required or color.get("legend") is not False
            self._append_axis_issues(encoding, issues)
            self._append_mark_scale_issues(unit, encoding, issues)
        return uses_color, visible_legend_required

    @staticmethod
    def _append_axis_issues(encoding: dict[str, Any], issues: list[ChartQualityIssue]) -> None:
        for channel_name in ("x", "y"):
            channel_def = encoding.get(channel_name)
            if not isinstance(channel_def, dict):
                continue
            axis = channel_def.get("axis")
            if axis is False:
                issues.append(ChartQualityIssue("position_axis_hidden", "critical", "position axis is hidden", {"channel": channel_name}))
            if isinstance(axis, dict) and not str(axis.get("title") or "").strip():
                issues.append(ChartQualityIssue("position_axis_title_missing", "warning", "position axis title is missing", {"channel": channel_name}))

    @staticmethod
    def _append_mark_scale_issues(unit: dict[str, Any], encoding: dict[str, Any], issues: list[ChartQualityIssue]) -> None:
        if mark_type(unit) != "bar":
            return
        for channel_name in ("x", "y"):
            channel_def = encoding.get(channel_name)
            if isinstance(channel_def, dict) and channel_def.get("type") == "quantitative":
                scale = channel_def.get("scale")
                if isinstance(scale, dict) and scale.get("zero") is False:
                    issues.append(ChartQualityIssue("bar_chart_truncated_axis", "critical", "bar chart uses a truncated quantitative axis", {"channel": channel_name}))

    def _scenegraph_metrics(self, scenegraph_summary: dict[str, Any] | None, issues: list[ChartQualityIssue]) -> dict[str, Any]:
        summary = scenegraph_summary or {}
        clipped_text_count = int(summary.get("clipped_text_count") or 0)
        plot_area_usage = float(summary.get("plot_area_usage") or 0.0)
        if clipped_text_count:
            issues.append(ChartQualityIssue("png_labels_clipped", "critical", "rendered chart has text outside canvas", {"clipped_text_count": clipped_text_count}))
        if plot_area_usage and plot_area_usage < self.thresholds.target_plot_area_usage:
            issues.append(ChartQualityIssue("low_plot_area_usage", "warning", "plot marks use too little of the rendered canvas", {"plot_area_usage": round(plot_area_usage, 4)}))
        return {
            "plot_area_usage": round(plot_area_usage, 4),
            "text_count": int(summary.get("text_count") or 0),
            "clipped_text_count": clipped_text_count,
            "mark_bbox_area": float(summary.get("mark_bbox_area") or 0.0),
        }

    def _dense_category_issues(self, spec: dict[str, Any], data: pd.DataFrame | None) -> list[ChartQualityIssue]:
        if data is None:
            return []
        issues: list[ChartQualityIssue] = []
        for unit in iter_unit_specs(spec):
            encoding = get_encoding(unit)
            for channel_name in ("x", "y"):
                issue = self._dense_category_issue(encoding.get(channel_name), channel_name, data)
                if issue is not None:
                    issues.append(issue)
        return issues

    def _dense_category_issue(self, channel_def: Any, channel_name: str, data: pd.DataFrame) -> ChartQualityIssue | None:
        if not isinstance(channel_def, dict) or channel_def.get("type") not in {"nominal", "ordinal"}:
            return None
        field = channel_def.get("field")
        if not isinstance(field, str) or field not in data.columns:
            return None
        labels = data[field].dropna().astype(str)
        if labels.empty:
            return None
        cardinality = int(labels.nunique())
        longest = int(labels.map(len).max())
        if channel_name == "x" and cardinality >= self.thresholds.dense_x_category_count and longest >= self.thresholds.dense_label_min_chars:
            return ChartQualityIssue(
                code="high_cardinality_x_categories",
                severity="warning",
                message="x-axis has many long categories; prefer top-N, ranking, or horizontal layout",
                details={"field": field, "cardinality": cardinality, "longest_label": longest},
            )
        return None

    @staticmethod
    def _png_size(path: Path) -> tuple[int, int]:
        with path.open("rb") as file:
            header = file.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
        raise ValueError(f"Rendered chart is not a PNG file: {path}")

    @staticmethod
    def _repeat_labels_are_specific(spec: dict[str, Any]) -> bool:
        repeated = ChartQualityEvaluator._repeat_fields(spec)
        if not repeated:
            return True
        visible_text = " ".join([title_text(spec.get("title")), ChartQualityEvaluator._axis_titles(spec)]).lower()
        return any(field.replace("_", " ").lower() in visible_text for field in repeated)

    @staticmethod
    def _repeat_fields(spec: dict[str, Any]) -> list[str]:
        repeat = spec.get("repeat")
        values: list[str] = []
        if isinstance(repeat, list):
            values.extend(str(value) for value in repeat if str(value).strip())
        if isinstance(repeat, dict):
            for key in ("row", "column"):
                field_values = repeat.get(key)
                if isinstance(field_values, list):
                    values.extend(str(value) for value in field_values if str(value).strip())
        return values

    @staticmethod
    def _axis_titles(spec: dict[str, Any]) -> str:
        titles: list[str] = []
        for unit in iter_unit_specs(spec):
            for channel_def in get_encoding(unit).values():
                if not isinstance(channel_def, dict):
                    continue
                axis = channel_def.get("axis")
                if isinstance(axis, dict):
                    title = axis.get("title")
                    if isinstance(title, str) and title.strip():
                        titles.append(title.strip())
        return " ".join(titles)

