from __future__ import annotations

from src.domain.enums import ChartCaseType
from src.services.planning_non_canonical import NonCanonicalPlanningService


def test_non_canonical_planning_service_builds_cautious_plan(
    non_canonical_query_understanding,
    runtime,
) -> None:
    service = NonCanonicalPlanningService()

    result = service.invoke(
        query_understanding=non_canonical_query_understanding,
        runtime=runtime,
    )

    assert result.mode is ChartCaseType.NON_CANONICAL
    assert result.steps
    assert any("assumption" in step.description.lower() or "simpler" in step.description.lower() for step in result.steps)
