from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.services.chart_quality._spec_utils import get_encoding, iter_unit_specs, mark_type
from src.services.chart_quality.chart_quality_types import ChartQualityIssue, ChartQualityReport


class ChartQualityEvaluator:
    HARD_FAIL_CODES = {
        "bar_chart_truncated_axis",
        "folded_metrics_shared_scale",
        "multi_metric_shared_scale_risk",
        "required_legend_not_rendered",
        "position_axis_hidden",
    }

    def __init__(self, *, hard_fail_codes: set[str] | None = None, max_warnings_for_pass: int = 2) -> None:
        if max_warnings_for_pass < 0:
            raise ValueError("max_warnings_for_pass must be non-negative.")
        self.hard_fail_codes = set(hard_fail_codes or self.HARD_FAIL_CODES)
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
        critical = sum(1 for issue in issues if issue.severity == "critical")
        warnings = sum(1 for issue in issues if issue.severity == "warning")
        status = self._status(issues=issues, critical=critical, warnings=warnings)
        score = max(0.0, 1.0 - critical * 0.4 - warnings * 0.08)
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
        metrics: dict[str, Any] = {}
        metrics.update(self._size_metrics(spec, png_path, issues))
        metrics.update(self._spec_metrics(spec, scenegraph_summary, issues))
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
    def _size_metrics(spec: dict[str, Any], png_path: str | Path | None, issues: list[ChartQualityIssue]) -> dict[str, Any]:
        logical_width = int(spec.get("width") or 0) if isinstance(spec.get("width"), int) else 0
        logical_height = int(spec.get("height") or 0) if isinstance(spec.get("height"), int) else 0
        pixel_width, pixel_height = ChartQualityEvaluator._png_size(Path(png_path)) if png_path else (0, 0)
        if pixel_width and pixel_height:
            ChartQualityEvaluator._append_canvas_issues(pixel_width=pixel_width, pixel_height=pixel_height, issues=issues)
        return {"logical_width": logical_width, "logical_height": logical_height, "pixel_width": pixel_width, "pixel_height": pixel_height}

    @staticmethod
    def _append_canvas_issues(*, pixel_width: int, pixel_height: int, issues: list[ChartQualityIssue]) -> None:
        aspect = pixel_width / max(pixel_height, 1)
        if aspect > 2.8:
            issues.append(ChartQualityIssue(
                code="excessive_wide_canvas",
                severity="warning",
                message="rendered image is very wide relative to height",
                details={"pixel_width": pixel_width, "pixel_height": pixel_height, "aspect_ratio": round(aspect, 3)},
            ))
        if aspect < 0.45:
            issues.append(ChartQualityIssue(
                code="excessive_tall_canvas",
                severity="warning",
                message="rendered image is very tall relative to width",
                details={"pixel_width": pixel_width, "pixel_height": pixel_height, "aspect_ratio": round(aspect, 3)},
            ))

    @staticmethod
    def _spec_metrics(spec: dict[str, Any], scenegraph_summary: dict[str, Any] | None, issues: list[ChartQualityIssue]) -> dict[str, Any]:
        unit_specs = list(iter_unit_specs(spec))
        uses_color, visible_legend_required = ChartQualityEvaluator._inspect_unit_specs(unit_specs, issues)
        has_legend = bool((scenegraph_summary or {}).get("has_legend"))
        repeat_used = bool(spec.get("repeat"))
        if uses_color and visible_legend_required and not has_legend:
            issues.append(ChartQualityIssue(code="required_legend_not_rendered", severity="critical", message="color encoding requires a visible legend"))
        if repeat_used and not ChartQualityEvaluator._title_exists(spec):
            issues.append(ChartQualityIssue(code="repeat_title_missing", severity="warning", message="repeat chart needs visible panel or chart title context"))
        return {"unit_spec_count": len(unit_specs), "uses_color": uses_color, "has_legend": has_legend, "repeat_used": repeat_used}

    @staticmethod
    def _inspect_unit_specs(unit_specs: list[dict[str, Any]], issues: list[ChartQualityIssue]) -> tuple[bool, bool]:
        uses_color = False
        visible_legend_required = False
        for unit in unit_specs:
            encoding = get_encoding(unit)
            color = encoding.get("color")
            if isinstance(color, dict) and color.get("field"):
                uses_color = True
                visible_legend_required = visible_legend_required or color.get("legend") is not False
            ChartQualityEvaluator._append_axis_issues(encoding, issues)
            ChartQualityEvaluator._append_mark_scale_issues(unit, encoding, issues)
        return uses_color, visible_legend_required

    @staticmethod
    def _append_axis_issues(encoding: dict[str, Any], issues: list[ChartQualityIssue]) -> None:
        for channel_name in ("x", "y"):
            channel_def = encoding.get(channel_name)
            if not isinstance(channel_def, dict):
                continue
            axis = channel_def.get("axis")
            if axis is False:
                issues.append(ChartQualityIssue(code="position_axis_hidden", severity="critical", message="position axis is hidden", details={"channel": channel_name}))
            if isinstance(axis, dict) and not str(axis.get("title") or "").strip():
                issues.append(ChartQualityIssue(code="position_axis_title_missing", severity="warning", message="position axis title is missing", details={"channel": channel_name}))

    @staticmethod
    def _append_mark_scale_issues(unit: dict[str, Any], encoding: dict[str, Any], issues: list[ChartQualityIssue]) -> None:
        if mark_type(unit) != "bar":
            return
        for channel_name in ("x", "y"):
            channel_def = encoding.get(channel_name)
            if isinstance(channel_def, dict) and channel_def.get("type") == "quantitative":
                scale = channel_def.get("scale")
                if isinstance(scale, dict) and scale.get("zero") is False:
                    issues.append(ChartQualityIssue(code="bar_chart_truncated_axis", severity="critical", message="bar chart uses a truncated quantitative axis", details={"channel": channel_name}))

    @staticmethod
    def _dense_category_issues(spec: dict[str, Any], data: pd.DataFrame | None) -> list[ChartQualityIssue]:
        if data is None:
            return []
        issues: list[ChartQualityIssue] = []
        for unit in iter_unit_specs(spec):
            encoding = get_encoding(unit)
            for channel_name in ("x", "y"):
                issue = ChartQualityEvaluator._dense_category_issue(encoding.get(channel_name), channel_name, data)
                if issue is not None:
                    issues.append(issue)
        return issues

    @staticmethod
    def _dense_category_issue(channel_def: Any, channel_name: str, data: pd.DataFrame) -> ChartQualityIssue | None:
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
        if channel_name == "x" and cardinality >= 18 and longest >= 10:
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
    def _title_exists(spec: dict[str, Any]) -> bool:
        title = spec.get("title")
        if isinstance(title, str):
            return bool(title.strip())
        if isinstance(title, dict):
            value = title.get("text")
            return bool(str(value or "").strip())
        return False
