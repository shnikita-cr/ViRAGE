from __future__ import annotations

from src.services.reasoner import ReasonerService


def test_reasoner_service_builds_insight_candidates_from_visual_facts(visual_facts_result, analysis_rubric, runtime) -> None:
    service = ReasonerService()
    result = service.invoke(visual_facts=visual_facts_result, analysis_rubric=analysis_rubric, runtime=runtime)
    assert result.insight_candidates
    assert "peak" in result.insight_candidates[0].statement.lower() or "sales" in result.insight_candidates[0].statement.lower()
