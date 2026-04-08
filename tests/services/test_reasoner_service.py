from __future__ import annotations

from src.services.reasoner import ReasonerService


def test_reasoner_service_builds_statements_from_facts(facts_result, runtime) -> None:
    service = ReasonerService()

    result = service.invoke(facts=facts_result, runtime=runtime)

    assert result.summary
    assert len(result.statements) == 2
    assert "chart type" in result.statements[0].text.lower()
