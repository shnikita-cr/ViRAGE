from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.domain.models import ImageOnlyChartJudgeResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured_multimodal
from src.services.base import BaseService


class _ImageOnlyChartJudgeSchema(BaseModel):
    chart_description: str = ""
    detected_chart_type: str | None = None
    visible_axes: dict[str, str] = Field(default_factory=dict)
    visible_legend: dict[str, Any] = Field(default_factory=dict)
    visible_labels: list[str] = Field(default_factory=list)
    is_chart_image: bool = True
    is_blank_or_unreadable: bool = False
    non_empty_score: float = 0.0
    readability_score: float = 0.0
    label_quality_score: float = 0.0
    legend_quality_score: float = 0.0
    visual_overload_score: float = 0.0
    plot_area_usage_score: float = 0.0
    axis_domain_score: float = 0.0
    layout_compactness_score: float = 0.0
    repeat_axis_label_score: float = 0.0
    publication_layout_score: float = 0.0
    confidence: float = 0.0
    detected_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    rationales: dict[str, str] = Field(default_factory=dict)


class ImageOnlyChartJudgeService(BaseService):
    """VLM judge for chart images when no query, data table, or Vega-Lite spec is available."""

    def invoke(self, *, image_path: str | Path, runtime: RuntimeContext) -> ImageOnlyChartJudgeResult:
        if runtime.vlm is None:
            raise RuntimeError("Image-only VLM benchmark requires runtime.vlm. No multimodal model was provided.")
        path = Path(image_path)
        parsed = invoke_structured_multimodal(
            runtime.vlm,
            self._prompt(),
            path.as_posix(),
            _ImageOnlyChartJudgeSchema,
            runtime=runtime,
            stage="vlm_image_benchmark",
            role="vlm",
            examples=[self._example()],
            max_attempts=2,
        )
        scores = {
            "non_empty_score": _clamp_score(parsed.non_empty_score),
            "readability_score": _clamp_score(parsed.readability_score),
            "label_quality_score": _clamp_score(parsed.label_quality_score),
            "legend_quality_score": _clamp_score(parsed.legend_quality_score),
            "visual_overload_score": _clamp_score(parsed.visual_overload_score),
            "plot_area_usage_score": _clamp_score(parsed.plot_area_usage_score),
            "axis_domain_score": _clamp_score(parsed.axis_domain_score),
            "layout_compactness_score": _clamp_score(parsed.layout_compactness_score),
            "repeat_axis_label_score": _clamp_score(parsed.repeat_axis_label_score),
            "publication_layout_score": _clamp_score(parsed.publication_layout_score),
        }
        overall = compute_overall_visual_score(scores)
        return ImageOnlyChartJudgeResult(
            chart_description=str(parsed.chart_description or "").strip(),
            detected_chart_type=parsed.detected_chart_type,
            visible_axes=dict(parsed.visible_axes or {}),
            visible_legend=dict(parsed.visible_legend or {}),
            visible_labels=_clean_list(parsed.visible_labels),
            is_chart_image=bool(parsed.is_chart_image),
            is_blank_or_unreadable=bool(parsed.is_blank_or_unreadable),
            **scores,
            overall_visual_score=overall,
            confidence=_clamp_score(parsed.confidence),
            detected_issues=_clean_list(parsed.detected_issues),
            recommendations=_clean_list(parsed.recommendations),
            rationales=dict(parsed.rationales or {}),
        )

    @staticmethod
    def _prompt() -> str:
        return (
            "You are ImageOnlyChartQualityJudgeAI. You receive only one chart image.\n"
            "Evaluate visible chart quality and publication readiness only. Do not judge data correctness, "
            "statistical correctness, field correctness, query alignment, or whether the chart answers a hidden user "
            "request, because no source table, prompt, or chart specification is provided.\n"
            "Assign all scores on a 0..1 scale, where 1 is best.\n"
            "Score definitions:\n"
            "- non_empty_score: visible chart is not blank and contains interpretable marks.\n"
            "- readability_score: axes, marks, titles, legends, and labels are readable.\n"
            "- label_quality_score: axis labels, panel headers, titles, and units are clear where visible.\n"
            "- legend_quality_score: legend is readable and useful; use 1.0 if no legend is needed.\n"
            "- visual_overload_score: chart is not overloaded; dense elements do not prevent interpretation.\n"
            "- plot_area_usage_score: data marks use the visible plot area efficiently without excessive empty regions.\n"
            "- axis_domain_score: visible axis ranges do not create excessive empty space or hide visible variation.\n"
            "- layout_compactness_score: canvas size/aspect ratio fits the amount of visual content.\n"
            "- repeat_axis_label_score: for repeated/faceted charts, panels and axes are clearly identified; use 1.0 if not applicable.\n"
            "- publication_layout_score: static figure could be used in a scientific report/paper without obvious cropping, resizing, or relabeling.\n"
            "List concrete detected_issues and actionable recommendations.\n"
            "Return only what can be judged from the image.\n"
        )

    @staticmethod
    def _example() -> dict[str, Any]:
        return {
            "chart_description": "A compact boxplot comparing three groups with readable labels.",
            "detected_chart_type": "boxplot",
            "visible_axes": {"x": "Group", "y": "Measurement"},
            "visible_legend": {},
            "visible_labels": ["Group", "Measurement"],
            "is_chart_image": True,
            "is_blank_or_unreadable": False,
            "non_empty_score": 0.95,
            "readability_score": 0.9,
            "label_quality_score": 0.9,
            "legend_quality_score": 1.0,
            "visual_overload_score": 0.9,
            "plot_area_usage_score": 0.85,
            "axis_domain_score": 0.8,
            "layout_compactness_score": 0.9,
            "repeat_axis_label_score": 1.0,
            "publication_layout_score": 0.85,
            "confidence": 0.9,
            "detected_issues": [],
            "recommendations": [],
            "rationales": {"scope": "Only visible image quality was evaluated."},
        }


def compute_overall_visual_score(scores: dict[str, float]) -> float:
    weights = {
        "non_empty_score": 0.15,
        "readability_score": 0.15,
        "label_quality_score": 0.10,
        "legend_quality_score": 0.05,
        "visual_overload_score": 0.10,
        "plot_area_usage_score": 0.15,
        "axis_domain_score": 0.10,
        "layout_compactness_score": 0.10,
        "repeat_axis_label_score": 0.05,
        "publication_layout_score": 0.05,
    }
    return round(sum(_clamp_score(scores.get(key, 0.0)) * weight for key, weight in weights.items()), 6)


def _clean_list(values: list[Any] | None) -> list[str]:
    return [str(item).strip() for item in (values or []) if str(item).strip()]


def _clamp_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(1.0, numeric))
