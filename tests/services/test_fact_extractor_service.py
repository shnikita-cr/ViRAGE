from __future__ import annotations

from src.services.fact_extractor import FactExtractorService


def test_fact_extractor_service_derives_visual_facts_from_vlm(vlm_analysis_result, runtime) -> None:
    service = FactExtractorService()
    result = service.invoke(vlm_analysis=vlm_analysis_result, runtime=runtime)
    assert result.visual_facts
    assert result.visual_facts[0].name.startswith("visual_fact_")
    assert result.evidence_refs
