from __future__ import annotations

from src.services.visrag import VisRAGService


def test_visrag_service_recommends_charts(
    canonical_query_understanding,
    sample_data_profile,
    runtime,
) -> None:
    service = VisRAGService()

    result = service.invoke(
        query_understanding=canonical_query_understanding,
        data_profile=sample_data_profile,
        runtime=runtime,
    )

    assert result.recommendations
    assert result.recommendations[0].chart_family == "line"
    assert result.rules
