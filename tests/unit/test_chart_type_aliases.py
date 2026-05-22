from __future__ import annotations

from src.services.spec_metrics.chart_type_utils import canonicalize_chart_type, normalize_aggregate


def test_canonicalize_chart_type_aliases() -> None:
    assert canonicalize_chart_type("scatter") == "point"
    assert canonicalize_chart_type("scatter plot") == "point"
    assert canonicalize_chart_type("grouped_bar") == "bar"
    assert canonicalize_chart_type("heat map") == "rect"


def test_normalize_aggregate_aliases() -> None:
    assert normalize_aggregate("avg") == "mean"
    assert normalize_aggregate("average") == "mean"
    assert normalize_aggregate("sum") == "sum"
