from __future__ import annotations

import pandas as pd

from src.domain.models import VegaLiteSpecArtifact
from src.services.spec.validator import SpecValidatorService


def test_scatter_mark_is_normalized_to_point(tmp_path):
    data_path = tmp_path / "points.csv"
    pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(data_path, index=False)
    spec = VegaLiteSpecArtifact(spec_json={
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"url": str(data_path)},
        "mark": "scatter",
        "encoding": {
            "x": {"field": "x", "type": "quantitative"},
            "y": {"field": "y", "type": "quantitative"},
        },
    })

    result = SpecValidatorService().invoke(spec)

    assert result.validated_spec["mark"] == "point"
    assert "Unsupported mark type" not in "\n".join(result.validation_errors)
