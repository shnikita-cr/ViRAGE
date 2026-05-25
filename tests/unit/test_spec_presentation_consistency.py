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
                "title": "Mean Value",
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
    assert "title" not in result.spec["encoding"]["y"]
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
                    "title": "Mean Value",
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

    assert result.spec["title"] == "Average Metrics by Model"
    assert "title" not in inner
    assert inner["encoding"]["x"]["axis"]["title"] == "Model"
    assert inner["encoding"]["y"]["axis"]["title"] == "Average Value"
    assert "title" not in inner["encoding"]["y"]
    assert inner["encoding"]["tooltip"][0]["title"] == "Average Value"
    assert inner["encoding"]["y"]["field"] == {"repeat": "column"}


def test_presentation_consistency_builds_boxplot_distribution_title_and_hides_duplicate_legend() -> None:
    spec = {
        "title": "PSNR by Method by Method",
        "mark": "boxplot",
        "encoding": {
            "x": {"field": "method", "type": "nominal"},
            "y": {"field": "psnr", "type": "quantitative"},
            "color": {"field": "method", "type": "nominal", "legend": {"title": "Method"}},
            "tooltip": {"field": "psnr", "aggregate": "median", "type": "quantitative", "title": "Median PSNR"},
        },
    }

    result = SpecPresentationConsistencyService().normalize(spec)

    assert result.spec["title"] == "PSNR Distribution by Method"
    assert result.spec["encoding"]["x"]["axis"]["title"] == "Method"
    assert result.spec["encoding"]["y"]["axis"]["title"] == "PSNR"
    assert result.spec["encoding"]["color"]["legend"] is None
    assert result.spec["encoding"]["tooltip"]["title"] == "PSNR"


def test_presentation_consistency_deduplicates_boxplot_group_dimensions() -> None:
    spec = {
        "mark": {"type": "boxplot"},
        "encoding": {
            "x": {"field": "gauss", "type": "nominal"},
            "y": {"field": "psnr", "type": "quantitative"},
            "color": {"field": "method", "type": "nominal"},
            "xOffset": {"field": "method", "type": "nominal"},
        },
    }

    result = SpecPresentationConsistencyService().normalize(spec)

    assert result.spec["title"] == "PSNR Distribution by Gauss and Method"


def test_presentation_consistency_uses_repeat_distribution_title_for_repeat_boxplots() -> None:
    spec = {
        "repeat": {"column": ["psnr", "ssim", "lpips"]},
        "spec": {
            "mark": "boxplot",
            "encoding": {
                "x": {"field": "method", "type": "nominal"},
                "y": {"field": {"repeat": "column"}, "type": "quantitative"},
            },
        },
    }

    result = SpecPresentationConsistencyService().normalize(spec)

    assert result.spec["title"] == "Metric Distributions by Method"
    assert result.spec["spec"]["encoding"]["y"]["axis"]["title"] == "Value"
