from __future__ import annotations

import pandas as pd

from src.services.chart_quality import ChartPresentationPolicy, ChartQualityEvaluator, ChartQualityPipeline, ChartSemanticPolicy
from src.services.rendering import ChartRenderPolicy


def test_bar_chart_keeps_zero_baseline() -> None:
    df = pd.DataFrame({"group": ["A", "B", "C"], "value": [90, 95, 99]})
    spec = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "group", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative", "scale": {"zero": False}},
        },
    }

    result = ChartSemanticPolicy().apply(spec, data=df)

    assert result.spec["encoding"]["y"]["scale"]["zero"] is True
    assert any(issue.code == "bar_chart_may_hide_small_differences" for issue in result.issues)


def test_local_domain_mark_can_disable_zero_for_close_values() -> None:
    df = pd.DataFrame({"group": ["A", "B", "C"], "value": [90, 95, 99]})
    spec = {
        "mark": "point",
        "encoding": {
            "x": {"field": "group", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
    }

    result = ChartSemanticPolicy().apply(spec, data=df)

    assert result.spec["encoding"]["y"]["scale"]["zero"] is False


def test_repeat_metrics_with_different_ranges_get_independent_scale_issue() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [100, 200, 300]})
    spec = {
        "repeat": {"column": ["a", "b"]},
        "spec": {
            "mark": "bar",
            "encoding": {
                "x": {"field": "category", "type": "nominal"},
                "y": {"field": {"repeat": "column"}, "type": "quantitative"},
            },
        },
    }

    result = ChartSemanticPolicy().apply(spec, data=df)

    assert result.spec["resolve"]["scale"]["y"] == "independent"
    assert any(issue.code == "multi_metric_shared_scale_risk" for issue in result.issues)


def test_presentation_policy_does_not_force_x_label_rotation_before_layout() -> None:
    df = pd.DataFrame({"file": [f"very_long_file_name_{i:02d}" for i in range(10)], "value": range(10)})
    spec = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "file", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
    }

    result = ChartPresentationPolicy().apply(spec, data=df)
    axis = result.spec["encoding"]["x"]["axis"]

    assert "labelAngle" not in axis
    assert axis["labelBound"] is True
    assert axis["labelOverlap"] == "greedy"


def test_render_policy_uses_padding_for_long_x_labels() -> None:
    df = pd.DataFrame({"file": [f"very_long_file_name_{i:02d}" for i in range(10)], "value": range(10)})
    spec = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "file", "type": "nominal"},
            "y": {"field": "value", "type": "quantitative"},
        },
    }

    pipeline = ChartQualityPipeline().apply(spec, data=df)
    rendered, policy = ChartRenderPolicy.apply(pipeline.spec, data=df, target="artifact", default_dpi=192)

    assert policy.padding["bottom"] >= 120
    assert rendered["padding"]["bottom"] == policy.padding["bottom"]


def test_quality_evaluator_rejects_hidden_position_axis() -> None:
    spec = {
        "width": 300,
        "height": 300,
        "mark": "bar",
        "encoding": {
            "x": {"field": "group", "type": "nominal", "axis": False},
            "y": {"field": "value", "type": "quantitative"},
        },
    }

    report = ChartQualityEvaluator().evaluate(spec=spec, scenegraph_summary={"has_legend": False})

    assert report.status in {"retry", "fail"}
    assert any(issue.code == "position_axis_hidden" for issue in report.issues)
