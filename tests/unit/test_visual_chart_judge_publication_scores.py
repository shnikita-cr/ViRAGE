from src.domain.feedback.visual_feedback_models import VisualChartJudgeResult
from src.services.visual_feedback.judges.visual_chart_judge import VisualChartJudgeAdapters, VisualChartJudgeService


def test_visual_chart_judge_prompt_requests_publication_scores() -> None:
    prompt = VisualChartJudgeService._prompt(
        query="compare groups",
        request_analysis=None,
        visual_judge_requirements={},
        use_chartsquared=False,
        chartsquared_max_eval_questions=4,
        prompt_max_chars=4000,
        chartsquared_project_root=None,
    )

    assert "plot_area_usage_score" in prompt
    assert "axis_domain_score" in prompt
    assert "layout_compactness_score" in prompt
    assert "repeat_axis_label_score" in prompt
    assert "publication_layout_score" in prompt
    assert "Tooltip-only evidence is not acceptable" in prompt


def test_visual_chart_judge_adapters_propagate_publication_issues() -> None:
    result = VisualChartJudgeResult(
        chart_description="A wide boxplot chart with excessive empty space.",
        answers_user_query=False,
        confidence=0.8,
        retry_recommendation="retry",
        layout_compactness_score=0.2,
        layout_compactness_issues=["Only three boxes are spread across a wide canvas."],
        repeat_axis_label_issues=["Facet headers do not identify the repeated metric."],
    )

    description = VisualChartJudgeAdapters.to_vlm_description(result)
    facts = VisualChartJudgeAdapters.to_fact_summary(result)
    judge = VisualChartJudgeAdapters.to_answer_judge(result)

    assert "Only three boxes are spread across a wide canvas." in description.readability_issues
    assert "Facet headers do not identify the repeated metric." in facts.quality_notes
    assert "Only three boxes are spread across a wide canvas." in judge.wrong_or_suspicious_parts
