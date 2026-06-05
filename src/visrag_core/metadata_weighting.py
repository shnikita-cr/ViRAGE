from __future__ import annotations

from dataclasses import dataclass

from src.domain.models import VisRAGGuidanceChunk


@dataclass(frozen=True)
class MetadataWeightingPolicy:
    manual_feedback_weight: float = 1.5
    scientific_figure_weight: float = 1.2
    min_weight: float = 0.1
    max_weight: float = 4.0

    def weight(self, chunk: VisRAGGuidanceChunk) -> float:
        priority = float((chunk.metadata or {}).get("priority", 1.0) or 1.0)
        weighted = priority * self._source_kind_multiplier(chunk.source_kind)
        upper = max(self.min_weight, self.max_weight)
        return max(self.min_weight, min(weighted, upper))

    def _source_kind_multiplier(self, source_kind: str) -> float:
        if source_kind in {"manual_feedback", "vlm_feedback"}:
            return self.manual_feedback_weight
        if source_kind == "scientific_figure_guidance":
            return self.scientific_figure_weight
        return 1.0

    def diagnostics(self) -> dict[str, float]:
        return {
            "manual_feedback_weight": self.manual_feedback_weight,
            "scientific_figure_weight": self.scientific_figure_weight,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
        }
