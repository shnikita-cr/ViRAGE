from __future__ import annotations

from src.domain.enums import ChartCaseType
from src.services.planning_canonical import CanonicalPlanningService


def test_canonical_planning_service_builds_fallback_plan(
    canonical_query_understanding,
    runtime,
) -> None:
    service = CanonicalPlanningService()

    result = service.invoke(
        query_understanding=canonical_query_understanding,
        runtime=runtime,
    )

    assert result.mode is ChartCaseType.CANONICAL
    assert result.steps
    assert any("chart" in step.description.lower() for step in result.steps)
