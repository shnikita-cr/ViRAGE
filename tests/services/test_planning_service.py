from __future__ import annotations

from src.services.planning import PlanningService


def test_planning_service_returns_default_pipeline_steps(
    canonical_query_understanding,
    runtime,
) -> None:
    service = PlanningService()

    result = service.invoke(
        query_understanding=canonical_query_understanding,
        runtime=runtime,
    )

    assert result.mode == canonical_query_understanding.case_type
    assert len(result.steps) == 6
    assert result.steps[0].name == "profile_data"
