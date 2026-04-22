from __future__ import annotations

from src.services.verifier import VerifierService


def test_verifier_service_marks_verified_visual_insights(insight_reasoning_result, runtime) -> None:
    service = VerifierService()
    result = service.invoke(insight_reasoning=insight_reasoning_result, runtime=runtime)
    assert result.all_verified is True
    assert result.verified_insights
    assert result.rejected_claims == []
