from __future__ import annotations

import pandas as pd

from src.domain.models import DataProfile, PlotImageArtifact, QueryUnderstandingResult, RequestAnalysisResult, SpecValidationResult, VegaLiteSpecArtifact
from src.services.chart_generator import ChartGeneratorService
from src.services.empty_chart_check import EmptyChartCheckService
from src.services.planning import PlanningService
from src.services.scenegraph_check import ScenegraphCheckService
from src.services.spec_validator import SpecValidatorService
from src.services.vegalite_plot_drawing import VegaLitePlotDrawingService
from src.services.visrag import VisRAGService


def test_rendering_flow_builds_non_empty_plot(canonical_query_understanding, request_analysis, sample_data_profile, prepared_result, runtime) -> None:
    visrag = VisRAGService().invoke(canonical_query_understanding, request_analysis, sample_data_profile, runtime=runtime)
    planning = PlanningService().invoke(canonical_query_understanding, request_analysis, sample_data_profile, visrag, runtime=runtime)
    generator = ChartGeneratorService()
    vega_spec = generator.invoke(prepared_result, visrag.candidate_spec_set, planning.execution_policy, planning.validation_policy, runtime=runtime)
    validation = SpecValidatorService().invoke(vega_spec)
    rendering = VegaLitePlotDrawingService().invoke(validation, run_id="render-test", runtime=runtime)
    scenegraph = ScenegraphCheckService().invoke(rendering)
    empty = EmptyChartCheckService().invoke(scenegraph)

    assert rendering.plot_image.image_path.endswith("plot.png")
    assert scenegraph.has_marks is True
    assert empty.non_empty_render is True


def test_rendering_flow_supports_count_by_category_without_y_field(tmp_path, runtime) -> None:
    data_path = tmp_path / "iris.csv"
    pd.DataFrame({"Species": ["setosa", "setosa", "virginica", "versicolor", "setosa"]}).to_csv(data_path, index=False)
    spec = VegaLiteSpecArtifact(
        spec_json={
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"url": data_path.as_posix()},
            "mark": "bar",
            "title": "Species distribution",
            "encoding": {
                "x": {"field": "Species", "type": "nominal"},
                "y": {"aggregate": "count", "type": "quantitative", "title": "Count"},
                "color": {"field": "Species", "type": "nominal", "legend": None},
            },
            "transform": [],
        },
        version="v1",
    )
    validation = SpecValidatorService().invoke(spec)
    assert validation.is_valid is True
    rendering = VegaLitePlotDrawingService().invoke(validation, run_id="render-iris-count", runtime=runtime)
    scenegraph = ScenegraphCheckService().invoke(rendering)
    empty = EmptyChartCheckService().invoke(scenegraph)

    assert rendering.plot_image.image_path.endswith("plot.png")
    assert scenegraph.has_marks is True
    assert empty.non_empty_render is True
