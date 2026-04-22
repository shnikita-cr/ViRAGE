from __future__ import annotations

from src.services.request_analyzer import RequestAnalyzerService


def test_request_analyzer_service_maps_query_to_fields(runtime, canonical_query_understanding, sample_data_profile) -> None:
    service = RequestAnalyzerService()

    result = service.invoke(
        query="Show sales trend over time by region",
        query_understanding=canonical_query_understanding,
        data_profile=sample_data_profile,
        runtime=runtime,
    )

    assert "date" in result.grounded_fields
    assert "sales" in result.grounded_fields
    assert "region" in result.selected_fields
    assert result.confidence > 0.0
