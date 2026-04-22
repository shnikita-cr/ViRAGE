from __future__ import annotations

from src.domain.models import RequestAnalysisResult
from src.services.planning import PlanningService


def test_planning_service_returns_execution_validation_and_analysis_policies(canonical_query_understanding,
                                                                             sample_data_profile, visrag_result,
                                                                             runtime) -> None:
    service = PlanningService()
    request_analysis = RequestAnalysisResult(
        grounded_fields=["date", "sales"],
        selected_fields=["date", "sales", "region"],
        ambiguity_report=[],
        normalization_hints=["parse date as datetime"],
        confidence=0.95,
    )

    result = service.invoke(
        query_understanding=canonical_query_understanding,
        request_analysis=request_analysis,
        data_profile=sample_data_profile,
        visrag=visrag_result,
        runtime=runtime,
    )

    assert result.mode is None
    assert result.execution_policy is not None
    assert result.validation_policy is not None
    assert result.analysis_rubric is not None
    assert result.steps
    assert result.analysis_rubric.strict_visual_only is True
