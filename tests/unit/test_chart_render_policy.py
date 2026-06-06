from __future__ import annotations

import pandas as pd

from src.services.rendering import ChartRenderPolicy


def test_render_policy_keeps_small_discrete_chart_compact_with_high_scale() -> None:
    df = pd.DataFrame({"group": ["A", "B", "C"], "value": [10, 20, 30]})
    spec = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "group", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
    }

    rendered, policy = ChartRenderPolicy.apply(spec, data=df, target="artifact", default_dpi=192)

    assert 280 <= policy.width < 620
    assert 260 <= policy.height <= 460
    assert policy.scale >= 2.0
    assert rendered["width"] == policy.width
    assert rendered["height"] == policy.height


def test_render_policy_allocates_more_width_for_many_categories() -> None:
    df = pd.DataFrame({"category": [f"category_{index}" for index in range(18)], "value": range(18)})
    spec = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
    }

    _, policy = ChartRenderPolicy.apply(spec, data=df, target="artifact", default_dpi=192)

    assert policy.width >= 700
    assert policy.width <= 860
    assert policy.scale >= 2.0


def test_render_policy_does_not_force_axis_label_angles() -> None:
    df = pd.DataFrame({"category": ["very long category label A", "very long category label B"], "value": [1, 2]})
    spec = {
        "mark": "bar",
        "encoding": {
            "y": {"field": "category", "type": "nominal"},
            "x": {"field": "value", "type": "quantitative"},
        },
    }

    rendered, _ = ChartRenderPolicy.apply(spec, data=df, target="artifact", default_dpi=192)

    axis_config = rendered["config"]["axis"]
    assert "labelAngle" not in axis_config
    assert axis_config["labelLimit"] >= 180


def test_render_policy_uses_publication_scale() -> None:
    df = pd.DataFrame({"x": range(20), "y": range(20)})
    spec = {
        "mark": "line",
        "encoding": {
            "x": {"field": "x", "type": "quantitative"},
            "y": {"field": "y", "type": "quantitative"},
        },
    }

    _, policy = ChartRenderPolicy.apply(spec, data=df, target="publication", default_dpi=192)

    assert policy.scale == 3.0


def test_render_policy_ignores_hidden_legends() -> None:
    df = pd.DataFrame({"group": ["A", "B", "C"] * 4, "value": list(range(12))})
    spec = {
        "mark": "point",
        "encoding": {
            "x": {"field": "group", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
            "color": {"field": "group", "type": "nominal", "legend": None},
        },
    }

    _, policy = ChartRenderPolicy.apply(spec, data=df, target="artifact", default_dpi=192)

    assert policy.width <= 430
    assert "legend_budget=0" in policy.reasoning


def test_render_policy_horizontal_categories_increase_height_not_width() -> None:
    df = pd.DataFrame({"category": [f"Category {index}" for index in range(10)], "score": range(10)})
    spec = {
        "mark": "bar",
        "encoding": {
            "y": {"field": "category", "type": "nominal"},
            "x": {"field": "score", "type": "quantitative", "aggregate": "mean"},
        },
    }

    rendered, policy = ChartRenderPolicy.apply(spec, data=df, target="artifact", default_dpi=192)

    assert policy.width <= 420
    assert policy.height >= 470
    assert rendered["padding"]["left"] >= 90


def test_render_policy_sets_nonzero_domain_for_distribution_marks() -> None:
    df = pd.DataFrame({"group": ["A", "B", "C"] * 5, "response": [90, 92, 94, 91, 93] * 3})
    spec = {
        "mark": "boxplot",
        "encoding": {
            "x": {"field": "group", "type": "nominal"},
            "y": {"field": "response", "type": "quantitative"},
        },
    }

    rendered, _ = ChartRenderPolicy.apply(spec, data=df, target="artifact", default_dpi=192)

    assert rendered["encoding"]["y"]["scale"]["zero"] is False


def test_render_policy_preserves_zero_domain_for_counts() -> None:
    df = pd.DataFrame({"value": [1, 2, 3, 4]})
    spec = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "value", "type": "quantitative", "bin": True},
            "y": {"aggregate": "count", "type": "quantitative"},
        },
    }

    rendered, _ = ChartRenderPolicy.apply(spec, data=df, target="artifact", default_dpi=192)

    assert rendered["encoding"]["y"]["scale"]["zero"] is True
    assert rendered["encoding"]["y"]["axis"]["tickMinStep"] == 1


def test_render_policy_protects_long_x_labels_with_padding() -> None:
    df = pd.DataFrame({"category": ["very long alpha category", "very long beta category"], "value": [1, 2]})
    spec = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
    }

    rendered, _ = ChartRenderPolicy.apply(spec, data=df, target="artifact", default_dpi=192)

    assert "labelAngle" not in rendered["encoding"]["x"]["axis"]
    assert 40 <= rendered["padding"]["bottom"] <= 80


def test_render_policy_filters_zero_missingness_fields_when_some_fields_have_missing_values() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, None, 3], "c": [7, 8, 9]})
    spec = {
        "transform": [
            {"fold": ["a", "b", "c"], "as": ["field", "value_check"]},
            {"calculate": "isValid(datum.value_check) ? 0 : 1", "as": "is_missing"},
            {"aggregate": [{"op": "mean", "field": "is_missing", "as": "missing_ratio"}], "groupby": ["field"]},
            {"calculate": "datum.missing_ratio * 100", "as": "missing_percentage"},
        ],
        "mark": "bar",
        "encoding": {
            "y": {"field": "field", "type": "nominal"},
            "x": {"field": "missing_percentage", "type": "quantitative"},
        },
    }

    rendered, policy = ChartRenderPolicy.apply(spec, data=df, target="artifact", default_dpi=192)

    assert rendered["transform"][-1] == {"filter": "datum.missing_percentage > 0"}
    assert "Filtered zero missing-percentage fields from missingness view." in policy.reasoning
