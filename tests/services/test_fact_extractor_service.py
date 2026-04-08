from __future__ import annotations

from src.services.fact_extractor import FactExtractorService


def test_fact_extractor_service_derives_facts_from_metrics(
    code_run_result,
    chart_read_result,
    runtime,
) -> None:
    service = FactExtractorService()

    result = service.invoke(
        execution=code_run_result,
        chart_read=chart_read_result,
        runtime=runtime,
    )

    fact_names = {fact.name for fact in result.facts}
    assert "chart_type" in fact_names
    assert "row_count" in fact_names
    assert "trend_hint" in fact_names
