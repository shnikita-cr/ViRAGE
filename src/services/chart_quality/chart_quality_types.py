from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class ChartQualityIssue:
    code: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "details": dict(self.details),
        }


def issue_dicts(issues: list["ChartQualityIssue"]) -> list[dict[str, Any]]:
    return [issue.to_dict() for issue in issues]


@dataclass(frozen=True)
class ChartPolicyResult:
    spec: dict[str, Any]
    issues: list[ChartQualityIssue] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)

    def issue_dicts(self) -> list[dict[str, Any]]:
        return issue_dicts(self.issues)


@dataclass(frozen=True)
class ChartQualityThresholds:
    max_aspect_ratio: float = 2.8
    min_aspect_ratio: float = 0.45
    max_rendered_width_px: int = 4800
    max_rendered_height_px: int = 3600
    dense_x_category_count: int = 18
    dense_label_min_chars: int = 10
    target_plot_area_usage: float = 0.35
    hard_fail_codes: tuple[str, ...] = (
        "render_failed",
        "empty_chart",
        "severity_field_not_used",
        "shared_axis_for_incompatible_metrics",
        "folded_metrics_shared_scale",
        "multi_metric_shared_scale_risk",
        "bar_chart_truncated_axis",
        "required_legend_not_rendered",
        "position_axis_hidden",
        "png_labels_clipped",
        "repeat_labels_invalid",
    )

    @classmethod
    def from_settings(cls, settings: Any) -> "ChartQualityThresholds":
        return cls(
            max_aspect_ratio=float(settings.chart_quality_max_aspect_ratio),
            min_aspect_ratio=float(settings.chart_quality_min_aspect_ratio),
            max_rendered_width_px=int(settings.chart_quality_max_rendered_width_px),
            max_rendered_height_px=int(settings.chart_quality_max_rendered_height_px),
            dense_x_category_count=int(settings.chart_quality_dense_x_category_count),
            dense_label_min_chars=int(settings.chart_quality_dense_label_min_chars),
            target_plot_area_usage=float(settings.chart_quality_target_plot_area_usage),
            hard_fail_codes=tuple(str(code) for code in settings.chart_quality_hard_fail_codes),
        )


@dataclass(frozen=True)
class ChartQualityReport:
    status: Literal["pass", "retry", "fail"]
    score: float
    issues: list[ChartQualityIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }
