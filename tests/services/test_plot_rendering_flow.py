from __future__ import annotations

from src.services.chart_generator import ChartGeneratorService
from src.services.empty_chart_check import EmptyChartCheckService
from src.services.planning import PlanningService
from src.services.scenegraph_check import ScenegraphCheckService
from src.services.spec_validator import SpecValidatorService
from src.services.vegalite_plot_drawing import VegaLitePlotDrawingService
from src.services.visrag import VisRAGService


def test_rendering_flow_builds_non_empty_plot(canonical_query_understanding, request_analysis, sample_data_profile,
                                              prepared_result, runtime) -> None:
    visrag = VisRAGService().invoke(canonical_query_understanding, request_analysis, sample_data_profile,
                                    runtime=runtime)
    planning = PlanningService().invoke(canonical_query_understanding, request_analysis, sample_data_profile, visrag,
                                        runtime=runtime)
    generator = ChartGeneratorService()
    vega_spec = generator.invoke(prepared_result, visrag.candidate_spec_set, planning.execution_policy,
                                 planning.validation_policy, runtime=runtime)
    validation = SpecValidatorService().invoke(vega_spec)
    rendering = VegaLitePlotDrawingService().invoke(validation, run_id="render-test", runtime=runtime)
    scenegraph = ScenegraphCheckService().invoke(rendering)
    empty = EmptyChartCheckService().invoke(scenegraph)

    assert rendering.plot_image.image_path.endswith("plot.png")
    assert scenegraph.has_marks is True
    assert empty.non_empty_render is True
