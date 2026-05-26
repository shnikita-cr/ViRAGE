from __future__ import annotations

import re
from pathlib import Path

from scripts.rag_corpus.common.text import compact_text

_VISUALIZATION_TERMS = {
    "chart", "charts", "plot", "plots", "visual", "visualization", "visualizations", "visualisation",
    "visualisations", "graph", "graphs", "axis", "axes", "legend", "legends", "tooltip", "tooltips",
    "label", "labels", "caption", "captions", "description", "descriptions", "mark", "marks",
    "encoding", "channel", "channels", "aggregate", "aggregation", "bin", "histogram", "scatter",
    "line", "bar", "map", "choropleth", "heatmap", "boxplot", "violin", "distribution",
    "correlation", "trend", "ranking", "comparison", "compare", "category", "categorical", "quantitative",
    "temporal", "time", "date", "color", "size", "facet", "facets", "sort", "filter", "scale",
    "readability", "accessibility", "contrast", "text", "summary", "summaries", "overplot", "outlier", "data",
}
_SOURCE_NOISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(load|install|import|require)\s+(the\s+)?(package|library|module|dependency|dependencies)\b",
        r"\b(pip|npm|yarn|conda|poetry|cargo|docker|webpack|vite)\b",
        r"\b(unit\s+test|test\s+suite|pytest|jest|coverage|ci|github\s+action)\b",
        r"\b(provider\s+tos|terms\s+of\s+service|license|copyright)\b",
        r"\b(api\s+key|token|authentication|authorization|login|account)\b",
        r"\b(html|css|javascript|typescript|python|r\s+code|script|function|class|method)\b.*\b(example|implementation|utility|helper)\b",
        r"\b(file|folder|directory|repository|repo|readme)\b.*\b(structure|organization|layout)\b",
        r"\b(release|changelog|contributing|contribution|issue|pull\s+request)\b",
    ]
]
_TITLE_NOISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^(installation|install|setup|getting started|quick start|usage|api|development|contributing|license|tests?|examples?)$",
        r"^(load required libraries|provider tos|organize .*scripts|name functions clearly|describe tests clearly)$",
    ]
]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()))


def is_relevant_visualization_source(
    *,
    title: str,
    text: str,
    path: Path | None = None,
    source_dataset: str = "",
    source_type: str = "",
    min_chars: int = 160,
) -> tuple[bool, str]:
    title_text = compact_text(title or "", max_chars=240)
    body = compact_text(text or "", max_chars=6000)
    combined = f"{title_text}. {body}".strip()
    if len(body) < min_chars:
        return False, "too_short_raw_text"
    if any(pattern.search(title_text.lower().strip()) for pattern in _TITLE_NOISE_PATTERNS):
        return False, "noise_title"
    tokens = _tokenize(combined)
    vis_hits = len(tokens & _VISUALIZATION_TERMS)
    if any(pattern.search(combined) for pattern in _SOURCE_NOISE_PATTERNS) and vis_hits < 4:
        return False, "documentation_or_code_noise"
    if vis_hits < 2:
        return False, "not_visualization_guidance"
    path_text = str(path or "").replace("\\", "/").lower()
    bad_path_parts = ("/test", "/tests", "/example", "/examples", "/demo", "/demos", "/doc/api", "/assets", "/static", "/css", "/js", "/scripts")
    if any(part in path_text for part in bad_path_parts) and vis_hits < 4:
        return False, "low_value_repository_area"
    return True, "kept"
