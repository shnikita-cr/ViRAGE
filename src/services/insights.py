from __future__ import annotations

from src.domain.models import InsightVerificationResult, InsightsResult
from src.services.base import BaseService


class InsightsService(BaseService):
    def invoke(self, insight_verification: InsightVerificationResult) -> InsightsResult:
        return InsightsResult(final_insights=list(insight_verification.verified_insights))
