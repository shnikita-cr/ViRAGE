from __future__ import annotations

import pandas as pd

from src.application.settings import ViRAGESettings
from src.domain.models import (
    CandidateSpec,
    CandidateSpecSet,
    DataPreparationResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.chart_generator import ChartGeneratorService
from src.services.spec_validator import SpecValidatorService


def test_chart_generator_does_not_add_axis_to_color_or_size(tmp_path):
    data_path = tmp_path / "iris.csv"
    pd.DataFrame(
        {
            "SepalLengthCm": [5.1, 4.9, 4.7],
            "SepalWidthCm": [3.5, 3.0, 3.2],
            "PetalLengthCm": [1.4, 1.4, 1.3],
            "PetalWidthCm": [0.2, 0.2, 0.2],
        }
    ).to_csv(data_path, index=False)

    candidate = CandidateSpec(
        spec_id="test:scatter_with_color_and_size",
        chart_family="point",
        summary="Create a scatter plot that shows the relationship between two quantitative fields.",
        rationale="Matched prepared RAG corpus example.",
        spec_template={
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "mark": "point",
            "encoding": {
                "x": {
                    "type": "quantitative",
                    "field": "SepalLengthCm",
                },
                "y": {
                    "type": "quantitative",
                    "field": "SepalWidthCm",
                },
                "color": {
                    "type": "quantitative",
                    "field": "PetalLengthCm",
                },
                "size": {
                    "type": "quantitative",
                    "field": "PetalWidthCm",
                },
            },
        },
    )
    candidate_set = CandidateSpecSet(
        candidate_specs=[candidate],
        selected_candidate_spec=candidate,
    )
    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
        )
    )

    artifact = ChartGeneratorService().invoke(
        prepared=DataPreparationResult(
            output_path=data_path.as_posix(),
            row_count=3,
            col_count=4,
        ),
        candidate_spec_set=candidate_set,
        runtime=runtime,
    )

    encoding = artifact.spec_json["encoding"]

    assert "axis" in encoding["x"]
    assert "axis" in encoding["y"]
    assert "axis" not in encoding["color"]
    assert "axis" not in encoding["size"]

    validation = SpecValidatorService().invoke(artifact)

    assert validation.is_valid, validation.validation_errors


def test_chart_generator_removes_existing_axis_from_non_position_channels(tmp_path):
    data_path = tmp_path / "iris.csv"
    pd.DataFrame(
        {
            "x": [1, 2],
            "y": [3, 4],
            "category": ["A", "B"],
        }
    ).to_csv(data_path, index=False)

    candidate = CandidateSpec(
        spec_id="test:bad_template_with_color_axis",
        chart_family="point",
        summary="Scatter with color.",
        spec_template={
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "mark": "point",
            "encoding": {
                "x": {
                    "type": "quantitative",
                    "field": "x",
                },
                "y": {
                    "type": "quantitative",
                    "field": "y",
                },
                "color": {
                    "type": "nominal",
                    "field": "category",
                    "axis": {
                        "labelLimit": 180,
                    },
                },
            },
        },
    )
    candidate_set = CandidateSpecSet(
        candidate_specs=[candidate],
        selected_candidate_spec=candidate,
    )
    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
        )
    )

    artifact = ChartGeneratorService().invoke(
        prepared=DataPreparationResult(
            output_path=data_path.as_posix(),
            row_count=2,
            col_count=3,
        ),
        candidate_spec_set=candidate_set,
        runtime=runtime,
    )

    assert "axis" not in artifact.spec_json["encoding"]["color"]

    validation = SpecValidatorService().invoke(artifact)

    assert validation.is_valid, validation.validation_errors
