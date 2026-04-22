from __future__ import annotations

from src.domain.models import SpecValidationResult
from src.services.spec_score import SpecScoreService


def test_spec_score_service_scores_validated_spec_not_raw_spec() -> None:
    service = SpecScoreService()
    validation = SpecValidationResult(
        validated_spec={
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"url": "dummy.csv"},
            "mark": "line",
            "title": "Sales trend",
            "encoding": {
                "x": {"field": "date", "type": "temporal"},
                "y": {"field": "sales", "type": "quantitative", "aggregate": "mean"},
            },
            "transform": [],
        },
        is_valid=True,
    )
    result = service.invoke(validation)
    assert result.score > 0.5
    assert "encoding present" in result.details


def test_spec_score_service_returns_zero_for_invalid_spec() -> None:
    service = SpecScoreService()
    result = service.invoke(SpecValidationResult(validated_spec={}, is_valid=False, validation_errors=["bad spec"]))
    assert result.score == 0.0
