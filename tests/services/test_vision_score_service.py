from __future__ import annotations

from src.services.vision_score import VisionScoreService


def test_vision_score_service_uses_weighted_criteria(plot_image, runtime) -> None:
    service = VisionScoreService()
    result = service.invoke(plot_image, runtime=runtime)
    assert result.score > 0.7
    assert any(detail.startswith("data_encoding=") for detail in result.details)
