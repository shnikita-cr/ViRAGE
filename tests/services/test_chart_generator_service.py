from __future__ import annotations

from src.services.chart_generator import ChartGeneratorService
from src.services.planning import PlanningService


def test_chart_generator_service_builds_vega_lite_spec(canonical_query_understanding, request_analysis, sample_data_profile, visrag_result, prepared_result, runtime) -> None:
    planning = PlanningService().invoke(
        query_understanding=canonical_query_understanding,
        request_analysis=request_analysis,
        data_profile=sample_data_profile,
        visrag=visrag_result,
        runtime=runtime,
    )
    service = ChartGeneratorService()
    result = service.invoke(
        prepared=prepared_result,
        candidate_spec_set=visrag_result.candidate_spec_set,
        execution_policy=planning.execution_policy,
        validation_policy=planning.validation_policy,
        runtime=runtime,
    )
    assert result.spec_json["mark"] == "line"
    assert result.spec_json["data"]["url"] == prepared_result.output_path
    assert result.spec_json["encoding"]["x"]["field"] == "date"
