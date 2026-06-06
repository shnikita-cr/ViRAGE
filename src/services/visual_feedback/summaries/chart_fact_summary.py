from __future__ import annotations

import json

from pydantic import BaseModel, Field

from src.domain.models import ChartFactSummaryResult, VLMChartDescriptionResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured
from src.services.base import BaseService


class _ChartFactSummarySchema(BaseModel):
    chart_type: str | None = None
    facts: list[str] = Field(default_factory=list)
    axes: dict[str, str] = Field(default_factory=dict)
    legend: dict = Field(default_factory=dict)
    visible_variables: list[str] = Field(default_factory=list)
    visible_relationships: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)


class ChartFactSummaryService(BaseService):
    """Summarize VLM description into facts without seeing the user query."""

    def invoke(self, description: VLMChartDescriptionResult, runtime: RuntimeContext) -> ChartFactSummaryResult:
        if runtime.reasoning_llm is None:
            raise RuntimeError("Chart fact summary requires runtime.reasoning_llm.")
        prompt = (
            "You are a chart fact extractor. Describe only facts visible in the chart image. Convert the chart description into structured facts.\n"
            "You must not use or ask for the user request. Use only this visual description.\n\n"
            f"VLM chart description JSON:\n{json.dumps(description.model_dump(), ensure_ascii=False, indent=2)}\n"
        )
        parsed = invoke_structured(
            runtime.reasoning_llm,
            prompt,
            _ChartFactSummarySchema,
            runtime=runtime,
            stage="chart_fact_summary",
            role="reasoner",
            examples=[{
                "chart_type": "bar",
                "facts": ["The chart compares categories using bar heights."],
                "axes": {"x": "categories", "y": "numeric values"},
                "legend": {},
                "visible_variables": ["categories", "numeric values"],
                "visible_relationships": ["category-to-value comparison"],
                "uncertainties": [],
                "quality_notes": [],
            }],
            max_attempts=2,
        )
        return ChartFactSummaryResult(**parsed.model_dump())
