from __future__ import annotations

from src.domain.models import PlotRenderingResult, ScenegraphCheckResult
from src.services.base import BaseService


class ScenegraphCheckService(BaseService):
    def invoke(self, plot_rendering: PlotRenderingResult) -> ScenegraphCheckResult:
        scenegraph = plot_rendering.rendered_scenegraph
        marks_count = int(scenegraph.get("marks_count", 0))
        axes = scenegraph.get("axes", [])
        has_legend = bool(scenegraph.get("has_legend", False))
        notes = list(scenegraph.get("notes", []))
        if marks_count <= 0:
            notes.append("No visual marks were detected in the rendered scenegraph.")
        return ScenegraphCheckResult(
            has_marks=marks_count > 0,
            has_axes=bool(axes),
            has_legends=has_legend,
            notes=notes,
        )
