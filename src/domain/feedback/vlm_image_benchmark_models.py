from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ImageOnlyChartJudgeResult(BaseModel):
    """Image-only VLM chart quality assessment.

    This model intentionally does not contain data-grounding or query-alignment fields: the benchmark receives only
    a chart image and can judge visible quality, not correctness against the source table or user request.
    """

    input_scope: Literal["image_only_chart_visual_quality"] = "image_only_chart_visual_quality"
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
    overall_visual_score: float = 0.0
    confidence: float = 0.0
    detected_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    rationales: dict[str, str] = Field(default_factory=dict)


class VLMImageBenchmarkImageResult(BaseModel):
    image_path: str
    relative_path: str
    file_name: str
    status: Literal["ok", "failed"] = "ok"
    error: str = ""
    result: ImageOnlyChartJudgeResult | None = None
    passed_publication_threshold: bool = False


class VLMImageBenchmarkSummary(BaseModel):
    run_id: str
    input_dir: str
    total_images: int
    evaluated_images: int
    failed_images: int
    publication_threshold: float
    pass_rate_publication_threshold: float
    mean_overall_visual_score: float | None = None
    median_overall_visual_score: float | None = None
    std_overall_visual_score: float | None = None
    mean_non_empty_score: float | None = None
    mean_readability_score: float | None = None
    mean_plot_area_usage_score: float | None = None
    mean_axis_domain_score: float | None = None
    mean_layout_compactness_score: float | None = None
    mean_repeat_axis_label_score: float | None = None
    mean_publication_layout_score: float | None = None
    issue_counts_by_type: dict[str, int] = Field(default_factory=dict)
    output_files: dict[str, str] = Field(default_factory=dict)
