from __future__ import annotations

from src.services.query_understanding import QueryUnderstandingService


def test_query_understanding_service_uses_reasoning_llm_and_returns_intent_bundle(runtime) -> None:
    service = QueryUnderstandingService()

    result = service.invoke(
        query="Show the sales trend over time",
        user_context={"max_charts": 1},
        runtime=runtime,
    )

    assert result.case_type is None
    assert result.task_type == "trend_analysis"
    assert result.user_goal == "understand sales movement over time"
    assert result.analysis_goal == "find trend shifts and peaks"
    assert "line" in result.candidate_charts
    assert result.query_variants
