from __future__ import annotations

from src.visrag_core import canonicalize_chart_type


def test_stacked_and_grouped_bar_chart_aliases_resolve_to_bar():
    assert canonicalize_chart_type("stacked_bar_chart") == "bar"
    assert canonicalize_chart_type("stacked bar chart") == "bar"
    assert canonicalize_chart_type("grouped_bar_chart") == "bar"
    assert canonicalize_chart_type("grouped bar chart") == "bar"
