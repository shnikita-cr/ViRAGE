from __future__ import annotations

from src.services.spec_presentation_consistency import SpecPresentationConsistencyService


def test_presentation_consistency_unifies_same_field_aggregate_labels() -> None:
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Average PSNR by Model",
        "mark": "bar",
        "encoding": {
            "x": {"field": "model", "type": "nominal"},
            "y": {
                "field": "psnr",
                "aggregate": "mean",
                "type": "quantitative",
                "axis": {"title": "Mean PSNR"},
            },
            "tooltip": [
                {
                    "field": "psnr",
                    "aggregate": "mean",
                    "type": "quantitative",
                    "title": "Average PSNR",
                }
            ],
        },
    }

    result = SpecPresentationConsistencyService().normalize(spec)

    assert result.spec["title"] == "Mean PSNR by Model"
    assert result.spec["encoding"]["y"]["axis"]["title"] == "Mean PSNR"
    assert result.spec["encoding"]["tooltip"][0]["title"] == "Mean PSNR"
    assert result.changes


def test_presentation_consistency_does_not_merge_different_fields() -> None:
    spec = {
        "mark": "point",
        "encoding": {
            "x": {"field": "psnr", "type": "quantitative", "axis": {"title": "PSNR"}},
            "y": {"field": "ssim", "type": "quantitative", "axis": {"title": "SSIM"}},
        },
    }

    result = SpecPresentationConsistencyService().normalize(spec)

    assert result.spec == spec
    assert result.changes == []
