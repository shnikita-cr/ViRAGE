from __future__ import annotations

from src.domain.models import RequestAnalysisResult
from src.services.planning_canonical import CanonicalPlanningService


def test_canonical_planning_service_delegates_to_unified_planning(canonical_query_understanding, sample_data_profile,
                                                                  visrag_result, runtime) -> None:
    service = CanonicalPlanningService()
    request_analysis = RequestAnalysisResult(
        grounded_fields=["date", "sales"],
        selected_fields=["date", "sales"],
        confidence=0.9,
    )

    result = service.invoke(canonical_query_understanding, request_analysis, sample_data_profile, visrag_result,
                            runtime)

    assert result.execution_policy is not None
    assert result.validation_policy is not None
