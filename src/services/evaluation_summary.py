from __future__ import annotations

from src.domain.models import EmptyChartCheckResult, EvaluationSummaryResult, InsightVerificationResult, StructuralSpecMetric, VisualQualityMetric
from src.services.base import BaseService


class EvaluationSummaryService(BaseService):
    def invoke(
        self,
        structural_spec_metric: StructuralSpecMetric,
        visual_quality_metric: VisualQualityMetric,
        empty_chart_check: EmptyChartCheckResult,
        insight_verification: InsightVerificationResult,
    ) -> EvaluationSummaryResult:
        report = {
            "spec_score": structural_spec_metric.score,
            "vision_score": visual_quality_metric.score,
            "empty_chart_status": empty_chart_check.empty_chart_status,
            "verified_insight_count": len(insight_verification.verified_insights),
            "rejected_claim_count": len(insight_verification.rejected_claims),
        }
        return EvaluationSummaryResult(
            structural_spec_metric=structural_spec_metric.score,
            visual_quality_metric=visual_quality_metric.score,
            empty_chart_status=empty_chart_check.empty_chart_status,
            insight_verification_summary=insight_verification.insight_verification_summary,
            benchmark_report=report,
        )
