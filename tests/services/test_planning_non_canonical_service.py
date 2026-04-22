from __future__ import annotations

from src.domain.models import QueryUnderstandingResult
from src.domain.models import RequestAnalysisResult
from src.services.planning_non_canonical import NonCanonicalPlanningService


def test_non_canonical_planning_service_delegates_to_unified_planning(sample_data_profile, visrag_result,
                                                                      runtime) -> None:
    service = NonCanonicalPlanningService()
    query_understanding = QueryUnderstandingResult(
        intent="Build a network diagram of flows",
        requested_operations=["relationship analysis"],
        candidate_charts=["scatter", "bar"],
        constraints=["prefer concise visuals"],
        confidence=0.66,
        task_type="relationship_analysis",
        user_goal="understand flow structure",
        analysis_goal="identify dominant relationships",
        ambiguity_notes=["network layout is underspecified"],
    )
    request_analysis = RequestAnalysisResult(
        grounded_fields=["source", "target", "flow_value"],
        selected_fields=["source", "target", "flow_value"],
        ambiguity_report=["node identifiers may require explicit source/target columns"],
        confidence=0.7,
    )

    result = service.invoke(query_understanding, request_analysis, sample_data_profile, visrag_result, runtime)

    assert result.execution_policy is not None
    assert result.analysis_rubric is not None
