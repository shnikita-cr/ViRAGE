from __future__ import annotations

from src.benchmark.evaluation.evaluator import VegaChatBenchmarkEvaluator
from src.domain.models import VisualQualityMetric
from src.services.evaluation.chart.vision_score import compute_vegachat_vision_score


def test_vegachat_vision_score_uses_expected_weights() -> None:
    score, weights = compute_vegachat_vision_score(
        visualization_type=2,
        data_encoding=1,
        data_transformation=2,
        aesthetics=0,
        prompt_compliance=2,
        is_blank=False,
    )

    expected = (1.0 * 1.0 + 0.5 * 2.0 + 1.0 * 1.0 + 0.0 * 0.75 + 1.0 * 1.5) / 6.25
    assert weights == {
        "visualization_type": 1.0,
        "data_encoding": 2.0,
        "data_transformation": 1.0,
        "aesthetics": 0.75,
        "prompt_compliance": 1.5,
    }
    assert score == round(expected, 6)


def test_vegachat_vision_score_applies_blank_penalty() -> None:
    score, _ = compute_vegachat_vision_score(
        visualization_type=2,
        data_encoding=2,
        data_transformation=2,
        aesthetics=2,
        prompt_compliance=2,
        is_blank=True,
    )

    assert 0.0 < score < 0.01


def test_benchmark_case_metrics_include_vision_submetrics() -> None:
    metric = VisualQualityMetric(
        score=0.8,
        visualization_type=1.0,
        data_encoding=0.5,
        data_transformation=1.0,
        aesthetics=0.5,
        prompt_compliance=1.0,
        is_blank=False,
    )

    metrics = VegaChatBenchmarkEvaluator._case_metrics(
        reference_spec={},
        generated_spec={},
        user_prompt="show data",
        is_valid=True,
        is_empty=False,
        spec_score=None,
        vision_score=metric.score,
        vision_is_blank=metric.is_blank,
        vision_metric=metric,
    )

    assert metrics["vision_score"] == 0.8
    assert metrics["vision_visualization_type"] == 1.0
    assert metrics["vision_data_encoding"] == 0.5
    assert metrics["vision_prompt_compliance"] == 1.0
    assert metrics["vision_is_empty_chart"] == 0.0
