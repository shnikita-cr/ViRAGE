from __future__ import annotations

from src.domain.models import DataProfile, PlanningResult, QueryUnderstandingResult, RequestAnalysisResult, VisRAGResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.planning import PlanningService


class NonCanonicalPlanningService(BaseService):
    def __init__(self) -> None:
        self._delegate = PlanningService()

    def invoke(
            self,
            query_understanding: QueryUnderstandingResult,
            request_analysis: RequestAnalysisResult,
            data_profile: DataProfile,
            visrag: VisRAGResult,
            runtime: RuntimeContext,
    ) -> PlanningResult:
        return self._delegate.invoke(query_understanding, request_analysis, data_profile, visrag, runtime)
