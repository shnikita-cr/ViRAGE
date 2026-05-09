from __future__ import annotations

from src.domain.models import EmptyChartCheckResult, EvaluationSummaryResult, InsightsResult, StructuralSpecMetric, VisualQualityMetric
from src.services.base import BaseService


class EvaluationSummaryService(BaseService):
    def invoke(
            self,
            structural_spec_metric: StructuralSpecMetric | None,
            visual_quality_metric: VisualQualityMetric | None,
            empty_chart_check: EmptyChartCheckResult,
            insights: InsightsResult | None,
    ) -> EvaluationSummaryResult:
        final_insights = insights.final_insights if insights else []
        report = {
            "spec_score": structural_spec_metric.score if structural_spec_metric else None,
            "vision_score": visual_quality_metric.score if visual_quality_metric else None,
            "empty_chart_status": empty_chart_check.empty_chart_status,
            "final_insight_count": len(final_insights),
        }
        return EvaluationSummaryResult(
            structural_spec_metric=report["spec_score"],
            visual_quality_metric=report["vision_score"],
            empty_chart_status=empty_chart_check.empty_chart_status,
            insight_summary="Insights come directly from reasoning output; verifier stage is removed.",
            benchmark_report=report,
        )
