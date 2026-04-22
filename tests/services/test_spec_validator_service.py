from __future__ import annotations

from src.domain.models import VegaLiteSpecArtifact
from src.services.spec_validator import SpecValidatorService


def test_spec_validator_accepts_valid_spec(prepared_result) -> None:
    service = SpecValidatorService()
    spec = VegaLiteSpecArtifact(
        spec_json={
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"url": prepared_result.output_path},
            "mark": "line",
            "encoding": {
                "x": {"field": "date", "type": "temporal"},
                "y": {"field": "sales", "type": "quantitative", "aggregate": "mean"},
            },
        }
    )
    result = service.invoke(spec)
    assert result.is_valid is True
    assert result.validation_errors == []
