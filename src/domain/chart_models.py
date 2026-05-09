from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VegaLiteSpecArtifact(BaseModel):
    spec_json: dict[str, Any] = Field(default_factory=dict)
    version: str | None = None
    generation_backend: str | None = None
    generation_explanation: str | None = None
    generation_warnings: list[str] = Field(default_factory=list)
    generation_artifacts: dict[str, str] = Field(default_factory=dict)


class SpecValidationResult(BaseModel):
    validated_spec: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    repair_hints: list[str] = Field(default_factory=list)
    is_valid: bool = False


class ScenegraphCheckResult(BaseModel):
    has_marks: bool = False
    has_axes: bool = False
    has_legends: bool = False
    notes: list[str] = Field(default_factory=list)


class EmptyChartCheckResult(BaseModel):
    empty_chart_signal: bool = False
    non_empty_render: bool = False
    empty_chart_status: str = "unknown"


class PlotImageArtifact(BaseModel):
    image_path: str
    width: int = 0
    height: int = 0


class PlotRenderingResult(BaseModel):
    plot_image: PlotImageArtifact
    rendered_scenegraph: dict[str, Any] = Field(default_factory=dict)
    render_notes: list[str] = Field(default_factory=list)
