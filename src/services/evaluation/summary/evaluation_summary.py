from __future__ import annotations

from typing import Any

from src.domain.models import EmptyChartCheckResult, EvaluationSummaryResult, InsightsResult, \
    SemanticFeedbackLoopSummary, StructuralSpecMetric
from src.services.base import BaseService


class EvaluationSummaryService(BaseService):
    def invoke(
            self,
            structural_spec_metric: StructuralSpecMetric | None,
            empty_chart_check: EmptyChartCheckResult,
            insights: InsightsResult | None,
            *,
            technical_status: str = "unknown",
            semantic_status: str = "unknown",
            semantic_summary: SemanticFeedbackLoopSummary | None = None,
            technical_retry_count: int = 0,
            benchmark_scores: dict[str, Any] | None = None,
    ) -> EvaluationSummaryResult:
        final_insights = insights.final_insights if insights else []
        insight_summary = self._insight_summary(final_insights)
        semantic_retry_count = int(getattr(semantic_summary, "retry_count", 0) or 0) if semantic_summary else 0
        semantic_issues = []
        if semantic_summary is not None:
            semantic_issues.extend(list(getattr(semantic_summary, "missing_requirements", []) or []))
            semantic_issues.extend(list(getattr(semantic_summary, "improvement_comments", []) or []))
        accepted = bool(getattr(semantic_summary, "accepted", False)) if semantic_summary else semantic_status in {
            "accepted", "disabled"}
        report = {
            "spec_score": structural_spec_metric.score if structural_spec_metric else None,
            "empty_chart_status": empty_chart_check.empty_chart_status,
            "technical_status": technical_status,
            "semantic_status": semantic_status,
            "chart_accepted": accepted and not bool(getattr(empty_chart_check, "empty_chart_signal", False)),
            "semantic_retry_count": semantic_retry_count,
            "technical_retry_count": technical_retry_count,
            "semantic_issue_count": len(semantic_issues),
            "final_insight_count": len(final_insights),
            **(benchmark_scores or {}),
        }
        return EvaluationSummaryResult(
            structural_spec_metric=report["spec_score"],
            empty_chart_status=empty_chart_check.empty_chart_status,
            visualization_error_rate_item=False if technical_status == "ok" else None,
            empty_chart_rate_item=bool(getattr(empty_chart_check, "empty_chart_signal", False)),
            technical_status=technical_status,
            semantic_status=semantic_status,
            chart_accepted=bool(report["chart_accepted"]),
            semantic_retry_count=semantic_retry_count,
            technical_retry_count=technical_retry_count,
            semantic_issues=semantic_issues,
            insight_summary=insight_summary,
            benchmark_scores=benchmark_scores or {"spec_score": report["spec_score"], "vision_score": None},
            benchmark_report=report,
        )
    @staticmethod
    def _insight_summary(final_insights: list[str]) -> str:
        cleaned = [item.strip() for item in final_insights if str(item).strip()]
        if not cleaned:
            return ""
        return "\n".join(f"- {item}" for item in cleaned[:5])

