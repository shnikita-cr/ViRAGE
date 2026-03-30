from src.domain.enums import ChartCaseType
from src.domain.models import DataProfile, QueryUnderstandingResult, VisRAGRecommendation, VisRAGResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class VisRAGService(BaseService):
    def invoke(self, query_understanding: QueryUnderstandingResult, data_profile: DataProfile,
               runtime: RuntimeContext) -> VisRAGResult:
        recommendations = []
        for idx, chart in enumerate(query_understanding.candidate_charts, start=1):
            rationale = f"Suggested for {', '.join(query_understanding.requested_operations)}."
            if chart == "line" and data_profile.likely_time_columns:
                rationale = "Time-like field detected; line chart suits trend analysis."
            if chart == "scatter" and len(data_profile.likely_numeric_columns) >= 2:
                rationale = "Two numeric fields detected; scatter suits relationship analysis."
            recommendations.append(VisRAGRecommendation(chart_family=chart, rationale=rationale, priority=idx))
        caveats = list(query_understanding.constraints)
        if query_understanding.case_type is ChartCaseType.NON_CANONICAL:
            caveats.append("Non-canonical case requires conservative interpretation.")
        return VisRAGResult(
            recommendations=recommendations,
            rules=["Use clear titles and axis labels.", "Avoid overcrowded visuals.", "Prefer readable defaults."],
            caveats=caveats,
        )
