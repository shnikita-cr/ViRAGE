from __future__ import annotations

from src.domain.models import EmptyChartCheckResult, InsightVerificationResult, StructuralSpecMetric, VisualQualityMetric
from src.services.evaluation_summary import EvaluationSummaryService


def test_evaluation_summary_service_aggregates_metrics() -> None:
    service = EvaluationSummaryService()
    result = service.invoke(
        structural_spec_metric=StructuralSpecMetric(score=0.8, details=["complete"]),
        visual_quality_metric=VisualQualityMetric(score=0.75, details=["clear"]),
        empty_chart_check=EmptyChartCheckResult(empty_chart_signal=False, non_empty_render=True, empty_chart_status="non_empty"),
        insight_verification=InsightVerificationResult(verified_insights=["Insight"], rejected_claims=[], insight_verification_summary="ok", all_verified=True),
    )
    assert result.benchmark_report["spec_score"] == 0.8
    assert result.empty_chart_status == "non_empty"
