from __future__ import annotations

from src.domain.enums import ChartCaseType
from src.services.query_understanding import QueryUnderstandingService


def test_query_understanding_service_routes_queries_without_llm(runtime) -> None:
    service = QueryUnderstandingService()

    canonical = service.invoke(
        query="Show the sales trend over time",
        user_context={"max_charts": 1},
        runtime=runtime,
    )
    non_canonical = service.invoke(
        query="Build a network diagram of flows between nodes",
        user_context={},
        runtime=runtime,
    )

    assert canonical.case_type is ChartCaseType.CANONICAL
    assert "line" in canonical.candidate_charts
    assert non_canonical.case_type is ChartCaseType.NON_CANONICAL
