from __future__ import annotations

from src.domain.models import VLMAnalysisResult, VisualFact, VisualFactExtractionResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class FactExtractorService(BaseService):
    def invoke(self, vlm_analysis: VLMAnalysisResult, runtime: RuntimeContext) -> VisualFactExtractionResult:
        facts = [
            VisualFact(name=f"visual_fact_{idx + 1}", value=value, evidence_refs=[f"observation:{idx + 1}"])
            for idx, value in enumerate(vlm_analysis.extracted_visual_facts)
        ]
        refs = [f"observation:{idx + 1}" for idx, _ in enumerate(vlm_analysis.visual_observations)]
        return VisualFactExtractionResult(visual_facts=facts, evidence_refs=refs)
