from __future__ import annotations

from src.services.spec_presentation_consistency import SpecPresentationConsistencyService


def test_presentation_consistency_builds_field_aware_aggregate_labels() -> None:
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Average by Model",
        "mark": "bar",
        "encoding": {
            "x": {"field": "model", "type": "nominal", "axis": {"title": "Category"}},
            "y": {
                "field": "psnr",
                "aggregate": "mean",
                "type": "quantitative",
                "axis": {"title": "Mean"},
            },
            "tooltip": [
                {
                    "field": "psnr",
                    "aggregate": "mean",
                    "type": "quantitative",
                    "title": "Avg",
                }
            ],
        },
    }

    result = SpecPresentationConsistencyService().normalize(spec)

    assert result.spec["title"] == "Average PSNR by Model"
    assert result.spec["encoding"]["x"]["axis"]["title"] == "Model"
    assert result.spec["encoding"]["y"]["axis"]["title"] == "Average PSNR"
    assert result.spec["encoding"]["tooltip"][0]["title"] == "Average PSNR"
    assert result.changes


def test_presentation_consistency_builds_time_series_title_with_grouping() -> None:
    spec = {
        "mark": "line",
        "encoding": {
            "x": {"field": "order_date", "type": "temporal", "timeUnit": "yearmonth"},
            "y": {"field": "sales", "aggregate": "mean", "type": "quantitative"},
            "color": {"field": "region", "type": "nominal"},
        },
    }

    result = SpecPresentationConsistencyService().normalize(spec)

    assert result.spec["title"] == "Average Sales over Year-Month by Region"
    assert result.spec["encoding"]["x"]["axis"]["title"] == "Year-Month"
    assert result.spec["encoding"]["y"]["axis"]["title"] == "Average Sales"
    assert result.spec["encoding"]["color"]["legend"]["title"] == "Region"


def test_presentation_consistency_does_not_merge_different_fields() -> None:
    spec = {
        "mark": "point",
        "encoding": {
            "x": {"field": "psnr", "type": "quantitative", "axis": {"title": "PSNR"}},
            "y": {"field": "ssim", "type": "quantitative", "axis": {"title": "SSIM"}},
        },
    }

    result = SpecPresentationConsistencyService().normalize(spec)

    assert result.spec["encoding"]["x"]["axis"]["title"] == "PSNR"
    assert result.spec["encoding"]["y"]["axis"]["title"] == "SSIM"
    assert result.spec["title"] == "SSIM vs PSNR"


def test_presentation_consistency_uses_repeat_safe_labels() -> None:
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "repeat": {"column": ["psnr", "ssim", "lpips"]},
        "spec": {
            "mark": "bar",
            "encoding": {
                "x": {"field": "model", "type": "nominal"},
                "y": {
                    "field": {"repeat": "column"},
                    "aggregate": "mean",
                    "type": "quantitative",
                    "axis": {"title": "Average PSNR"},
                },
                "tooltip": [
                    {
                        "field": {"repeat": "column"},
                        "aggregate": "mean",
                        "type": "quantitative",
                        "title": "Average PSNR",
                    }
                ],
            },
        },
    }

    result = SpecPresentationConsistencyService().normalize(spec)
    inner = result.spec["spec"]

    assert inner["title"] == "Average Metrics by Model"
    assert inner["encoding"]["x"]["axis"]["title"] == "Model"
    assert inner["encoding"]["y"]["axis"]["title"] == "Average Value"
    assert inner["encoding"]["tooltip"][0]["title"] == "Average Value"
    assert inner["encoding"]["y"]["field"] == {"repeat": "column"}
