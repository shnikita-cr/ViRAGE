from __future__ import annotations

from typing import Any

from src.domain.models import EmptyChartCheckResult, SpecValidationResult, StructuralSpecMetric
from src.services.base import BaseService
from src.services.vegachat_spec_metrics import compute_vegachat_spec_score


class SpecScoreService(BaseService):
    """VegaChat Spec Score for generated Vega-Lite specifications.

    The formula mirrors VegaChat's deterministic Spec Score:
    - a non-drawable chart receives 0;
    - valid schema and drawable status receive a small positive contribution;
    - empty charts receive a very large penalty in the denominator;
    - marks are scored with F1 and partial equivalence for circle/point/square;
    - encodings are scored with weighted F-beta, beta=2, x/y and row/column swaps allowed;
    - view-level transforms are scored with F1 over normalized transform paths;
    - unnecessary view-level transforms penalize the encoding component.
    """

    def invoke(
        self,
        spec_validation: SpecValidationResult,
        ground_truth_spec: dict[str, Any],
        *,
        user_prompt: str | None = None,
        empty_chart_check: EmptyChartCheckResult | None = None,
    ) -> StructuralSpecMetric:
        generated_spec = spec_validation.validated_spec or {}
        is_drawable = bool(spec_validation.is_valid)
        is_valid_schema = bool(spec_validation.is_valid)
        is_empty = _is_empty(empty_chart_check)

        if not generated_spec:
            return StructuralSpecMetric(
                score=0.0,
                validity_score=0.0,
                empty_chart_penalty=0.0,
                details=["formula=vegachat_spec_score_impl", "generated_spec_missing"],
            )

        result = compute_vegachat_spec_score(
            ground_truth_spec,
            generated_spec,
            utterance=user_prompt or "",
            hyp_is_drawable=is_drawable,
            hyp_is_empty_chart=is_empty,
            hyp_is_valid_schema=is_valid_schema,
        )
        metrics = result.to_metrics_dict()
        empty_penalty = 0.0 if is_empty else 1.0
        return StructuralSpecMetric(
            score=round(result.spec_score, 6),
            mark_score=round(result.mark.f1, 6),
            encoding_score=round(result.encoding.f1, 6),
            transform_score=round(result.transform.f1, 6),
            validity_score=1.0 if is_valid_schema else 0.0,
            empty_chart_penalty=empty_penalty,
            encoding_precision=round(result.encoding.precision, 6),
            encoding_recall=round(result.encoding.recall, 6),
            transform_precision=round(result.transform.precision, 6),
            transform_recall=round(result.transform.recall, 6),
            mark_precision=round(result.mark.precision, 6),
            mark_recall=round(result.mark.recall, 6),
            weights={
                "drawable": 0.005,
                "valid_schema": 0.005 if is_valid_schema else 1.0,
                "not_empty": 1000.0 if is_empty else 0.005,
                "encoding": 3.0,
                "mark": 1.0 if _prompt_mentions_mark(user_prompt or "") else 0.5,
                "transform": 1.0,
            },
            details=[
                *result.details,
                *[f"{key}={value:.6f}" for key, value in metrics.items() if isinstance(value, float)],
            ],
        )


def _is_empty(empty_chart_check: EmptyChartCheckResult | None) -> bool:
    if empty_chart_check is None:
        return False
    return bool(empty_chart_check.empty_chart_signal or empty_chart_check.empty_chart_status == "empty")


def _prompt_mentions_mark(prompt: str) -> bool:
    from src.services.vegachat_spec_metrics import get_marks_in_utterance

    return bool(get_marks_in_utterance(prompt))
