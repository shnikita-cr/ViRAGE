from __future__ import annotations

import pandas as pd

from src.domain.models import VegaLiteSpecArtifact
from src.services.spec_validator import SpecValidatorService


def test_spec_validator_accepts_simple_valid_spec(tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "sales": [10, 20]}).to_csv(data_path, index=False)
    spec = VegaLiteSpecArtifact(
        spec_json={
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"url": data_path.as_posix()},
            "mark": "line",
            "encoding": {
                "x": {"field": "date", "type": "temporal"},
                "y": {"field": "sales", "type": "quantitative"},
            },
        },
        version="v1",
    )

    result = SpecValidatorService().invoke(spec)

    assert result.is_valid is True
    assert result.validation_errors == []


def test_spec_validator_accepts_count_without_y_field(tmp_path) -> None:
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"Species": ["setosa", "setosa", "virginica"]}).to_csv(data_path, index=False)
    spec = VegaLiteSpecArtifact(
        spec_json={
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"url": data_path.as_posix()},
            "mark": "bar",
            "encoding": {
                "x": {"field": "Species", "type": "nominal"},
                "y": {"aggregate": "count", "type": "quantitative", "title": "Count"},
            },
        },
        version="v1",
    )

    result = SpecValidatorService().invoke(spec)

    assert result.is_valid is True
    assert result.validation_errors == []
