from __future__ import annotations

from src.services.vlm_analysis import VLMAnalysisService


def test_vlm_analysis_service_returns_visual_observations(plot_image, analysis_rubric, runtime) -> None:
    service = VLMAnalysisService()
    result = service.invoke(plot_image=plot_image, analysis_rubric=analysis_rubric, runtime=runtime)
    assert result.visual_observations
    assert result.extracted_visual_facts
