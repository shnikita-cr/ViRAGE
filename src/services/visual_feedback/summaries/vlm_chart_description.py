from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import PlotImageArtifact, VLMChartDescriptionResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured_multimodal
from src.services.base import BaseService


class _VLMChartDescriptionSchema(BaseModel):
    visual_description: str = ""
    detected_chart_type: str | None = None
    visible_axes: dict[str, str] = Field(default_factory=dict)
    visible_legend: dict = Field(default_factory=dict)
    visible_labels: list[str] = Field(default_factory=list)
    visible_trends: list[str] = Field(default_factory=list)
    visible_comparisons: list[str] = Field(default_factory=list)
    visible_outliers: list[str] = Field(default_factory=list)
    readability_issues: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class VLMChartDescriptionService(BaseService):
    """Describe only the rendered PNG. No query/spec/data context is passed."""

    def invoke(self, plot_image: PlotImageArtifact, runtime: RuntimeContext) -> VLMChartDescriptionResult:
        if runtime.vlm is None:
            raise RuntimeError("Semantic VLM loop requires runtime.vlm. No multimodal model was provided.")

        prompt = (
            "You are a chart-image description expert. Summarize the chart content without inventing source data. Analyze only the attached chart image.\n"
            "You do not have access to the user request, Vega-Lite spec, source table, or data profile.\n"
            "Do not infer anything that is not visible in the image.\n"
            "Describe the chart as completely as possible: chart type, visible axes, legends, labels, trends, "
            "comparisons, outliers, readability problems, and uncertainties.\n"
        )
        parsed = invoke_structured_multimodal(
            runtime.vlm,
            prompt,
            plot_image.image_path,
            _VLMChartDescriptionSchema,
            runtime=runtime,
            stage="vlm_chart_description",
            role="vlm",
            examples=[{
                "visual_description": "The image shows a bar chart with categories on the x-axis and a numeric y-axis.",
                "detected_chart_type": "bar",
                "visible_axes": {"x": "category labels", "y": "numeric values"},
                "visible_legend": {},
                "visible_labels": ["category labels", "numeric y-axis"],
                "visible_trends": [],
                "visible_comparisons": ["Bars compare category heights."],
                "visible_outliers": [],
                "readability_issues": [],
                "uncertainties": [],
                "confidence": 0.8,
            }],
            max_attempts=2,
        )
        return VLMChartDescriptionResult(input_scope="png_only", **parsed.model_dump())
