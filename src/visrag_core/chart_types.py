from __future__ import annotations

_CHART_ALIASES = {
    "scatter": "point",
    "scatterplot": "point",
    "scatter_plot": "point",
    "scatter plot": "point",
    "bar chart": "bar",
    "line chart": "line",
    "area chart": "area",
    "histogram chart": "histogram",
}

SUPPORTED_CHART_TYPES = {"bar", "line", "area", "point", "circle", "tick", "histogram", "boxplot"}


def canonicalize_chart_type(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_")
    if not text:
        return ""
    return _CHART_ALIASES.get(text, text)


def require_supported_chart_type(value: str | None) -> str:
    chart_type = canonicalize_chart_type(value)
    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ValueError(f"Unsupported chart type: {value!r}")
    return chart_type


def normalize_aggregate(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return "mean" if text in {"avg", "average"} else text
