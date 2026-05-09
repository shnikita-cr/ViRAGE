from __future__ import annotations

from src.domain.models import InsightReasoningResult, InsightsResult
from src.services.base import BaseService


class InsightsService(BaseService):
    def invoke(self, insight_reasoning: InsightReasoningResult) -> InsightsResult:
        return InsightsResult(
            final_insights=[candidate.statement for candidate in insight_reasoning.insight_candidates]
        )
