from __future__ import annotations

_CHART_ALIASES = {
    "scatter": "point",
    "scatterplot": "point",
    "scatter_plot": "point",
    "scatter plot": "point",
    "bar chart": "bar",
    "bar_chart": "bar",
    "stacked_bar": "bar",
    "stacked_bar_chart": "bar",
    "stacked bar": "bar",
    "stacked bar chart": "bar",
    "grouped_bar": "bar",
    "grouped_bar_chart": "bar",
    "grouped bar": "bar",
    "grouped bar chart": "bar",
    "column": "bar",
    "column_chart": "bar",
    "line chart": "line",
    "line_chart": "line",
    "area chart": "area",
    "area_chart": "area",
    "hist": "histogram",
    "histogram chart": "histogram",
    "box": "boxplot",
    "box_plot": "boxplot",
    "heatmap": "rect",
    "heat_map": "rect",
    "matshow": "rect",
    "imshow": "rect",
}

SUPPORTED_CHART_TYPES = {
    "bar",
    "line",
    "area",
    "point",
    "circle",
    "square",
    "tick",
    "histogram",
    "boxplot",
    "rect",
    "rule",
    "text",
}


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
