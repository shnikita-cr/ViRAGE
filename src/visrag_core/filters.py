from __future__ import annotations

import re
from typing import Any

from src.domain.models import DataProfile, QueryRequestAnalysisResult, VisRAGRuleDocument


def domain_semantics_gate(query_analysis: QueryRequestAnalysisResult, data_profile: DataProfile) -> bool:
    text = " ".join([
        query_analysis.normalized_query,
        " ".join(query_analysis.selected_fields),
        " ".join(column.name for column in data_profile.columns),
    ]).lower()
    domain_markers = {
        "biology", "biological", "medical", "clinical", "gene", "protein", "species",
        "hba1c", "crp", "ast", "alt", "bmi", "gdp", "inflation", "population",
        "finance", "economic", "economics", "ecology", "treatment", "diagnosis",
    }
    return any(marker in text for marker in domain_markers)


_MAP_MARKERS = {
    "map", "choropleth", "geo", "geographic", "geographical", "cartogram",
    "latitude", "longitude", "geospatial", "region shape", "geometry",
}
_NETWORK_MARKERS = {"network", "node", "edge", "source", "target", "connection", "link"}
_WORDCLOUD_MARKERS = {"wordcloud", "word cloud"}
_LINE_MARKERS = {"line", "time", "trend", "temporal", "time series", "date"}
_SCATTER_MARKERS = {"scatter", "correlation", "relationship", "two quantitative"}
_BAR_MARKERS = {"bar", "column", "rank", "category", "categorical", "compare"}
_HISTOGRAM_MARKERS = {"histogram", "distribution", "density", "boxplot", "violin"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | tuple | set):
        return " ".join(_normalize_text(item) for item in value)
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _doc_text(doc: VisRAGRuleDocument) -> str:
    metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
    parts = [
        doc.doc_id,
        doc.title,
        doc.prompt_text,
        doc.retrieval_text,
        metadata.get("chart_family"),
        metadata.get("task"),
        metadata.get("applies_when"),
        metadata.get("guidance"),
        metadata.get("avoid"),
    ]
    return _normalize_text(parts)


def _query_text(query_analysis: QueryRequestAnalysisResult) -> str:
    parts = [
        query_analysis.normalized_query,
        query_analysis.analysis_task,
        query_analysis.recommended_chart_family,
        query_analysis.selected_fields,
        query_analysis.aggregation_plan,
    ]
    return _normalize_text(parts)


def _has_any(text: str, markers: set[str]) -> bool:
    return any(marker in text for marker in markers)


def _data_has_geo(data_profile: DataProfile) -> bool:
    names = " ".join(column.name for column in data_profile.columns).lower()
    roles = " ".join(column.role for column in data_profile.columns).lower()
    geo_terms = {
        "latitude", "longitude", "lat", "lon", "lng", "geometry", "geom", "geojson",
        "country", "state", "province", "county", "city", "zip", "postal", "fips",
    }
    return any(term in names for term in geo_terms) or "geo" in roles or "geographic" in roles


def _data_has_network_fields(data_profile: DataProfile) -> bool:
    names = {column.name.lower() for column in data_profile.columns}
    joined = " ".join(names)
    return (
        {"source", "target"}.issubset(names)
        or {"from", "to"}.issubset(names)
        or ("origin" in joined and "destination" in joined)
    )


def _data_has_text_field(data_profile: DataProfile) -> bool:
    return any(
        column.role.lower() in {"text", "description"}
        or "text" in column.dtype.lower()
        or column.name.lower() in {"word", "term", "token", "text", "description"}
        for column in data_profile.columns
    )


def _target_family(query_analysis: QueryRequestAnalysisResult) -> str:
    return _normalize_text(query_analysis.recommended_chart_family)


def _target_task(query_analysis: QueryRequestAnalysisResult) -> str:
    return _normalize_text(query_analysis.analysis_task)


def incompatibility_reason(
        doc: VisRAGRuleDocument,
        query_analysis: QueryRequestAnalysisResult,
        data_profile: DataProfile,
) -> str | None:
    """Return a reason when a retrieved rule contradicts the current chart request.

    The filter is intentionally conservative. It removes only high-risk chart-pattern
    rules that require a data shape the current table/query does not provide.
    """
    text = _doc_text(doc)
    target = _target_family(query_analysis)
    task = _target_task(query_analysis)
    query = _query_text(query_analysis)

    wants_line_or_trend = _has_any(" ".join([target, task, query]), _LINE_MARKERS) or task == "trend"
    wants_scatter = _has_any(" ".join([target, task, query]), _SCATTER_MARKERS)
    wants_bar = _has_any(" ".join([target, task, query]), _BAR_MARKERS)
    wants_distribution = _has_any(" ".join([target, task, query]), _HISTOGRAM_MARKERS)
    wants_map = _has_any(" ".join([target, task, query]), _MAP_MARKERS)
    wants_network = _has_any(" ".join([target, task, query]), _NETWORK_MARKERS)
    wants_wordcloud = _has_any(" ".join([target, task, query]), _WORDCLOUD_MARKERS)

    if _has_any(text, _MAP_MARKERS) and not wants_map:
        return "incompatible_chart_family:map_rule_for_non_map_request"
    if _has_any(text, _MAP_MARKERS) and not _data_has_geo(data_profile):
        return "missing_required_data:geographic_fields"
    if _has_any(text, _NETWORK_MARKERS) and not wants_network and doc.record_type == "chart_pattern":
        return "incompatible_chart_family:network_rule_for_non_network_request"
    if _has_any(text, _NETWORK_MARKERS) and not _data_has_network_fields(data_profile) and doc.record_type == "chart_pattern":
        return "missing_required_data:source_target_fields"
    if _has_any(text, _WORDCLOUD_MARKERS) and not wants_wordcloud:
        return "incompatible_chart_family:wordcloud_rule_for_non_text_request"
    if _has_any(text, _WORDCLOUD_MARKERS) and not _data_has_text_field(data_profile):
        return "missing_required_data:text_field"

    if wants_line_or_trend and doc.record_type == "chart_pattern" and _has_any(text, _BAR_MARKERS | _SCATTER_MARKERS | _HISTOGRAM_MARKERS) and not _has_any(text, _LINE_MARKERS):
        return "incompatible_chart_family:non_line_rule_for_trend_request"
    if wants_scatter and doc.record_type == "chart_pattern" and _has_any(text, _BAR_MARKERS | _HISTOGRAM_MARKERS) and not _has_any(text, _SCATTER_MARKERS):
        return "incompatible_chart_family:non_scatter_rule_for_relationship_request"
    if wants_distribution and doc.record_type == "chart_pattern" and _has_any(text, _BAR_MARKERS | _LINE_MARKERS | _SCATTER_MARKERS) and not _has_any(text, _HISTOGRAM_MARKERS):
        return "incompatible_chart_family:non_distribution_rule_for_distribution_request"
    if wants_bar and doc.record_type == "chart_pattern" and _has_any(text, _LINE_MARKERS | _SCATTER_MARKERS | _HISTOGRAM_MARKERS) and not _has_any(text, _BAR_MARKERS):
        return "incompatible_chart_family:non_bar_rule_for_category_comparison"
    return None


def compatibility_bonus(doc: VisRAGRuleDocument, query_analysis: QueryRequestAnalysisResult) -> float:
    text = _doc_text(doc)
    target_task_text = " ".join([_target_family(query_analysis), _target_task(query_analysis), _query_text(query_analysis)])
    bonus = 0.0
    marker_groups = [_LINE_MARKERS, _SCATTER_MARKERS, _BAR_MARKERS, _HISTOGRAM_MARKERS, _MAP_MARKERS]
    for markers in marker_groups:
        if _has_any(target_task_text, markers) and _has_any(text, markers):
            bonus += 25.0
    aggregation_text = _normalize_text(query_analysis.aggregation_plan)
    if aggregation_text and any(token in text for token in ["mean", "average", "sum", "count", "aggregate", "group"]):
        bonus += 10.0
    return bonus


def rerank_by_compatibility(
        ranked: list[VisRAGRuleDocument],
        query_analysis: QueryRequestAnalysisResult,
        data_profile: DataProfile,
) -> tuple[list[VisRAGRuleDocument], list[dict[str, Any]]]:
    kept: list[VisRAGRuleDocument] = []
    filtered: list[dict[str, Any]] = []
    for doc in ranked:
        reason = incompatibility_reason(doc, query_analysis, data_profile)
        if reason:
            filtered.append({
                "doc_id": doc.doc_id,
                "record_type": doc.record_type,
                "reason": reason,
                "score": doc.score,
            })
            continue
        bonus = compatibility_bonus(doc, query_analysis)
        if bonus and hasattr(doc, "model_copy"):
            doc = doc.model_copy(update={"score": float(doc.score or 0.0) + bonus})
        kept.append(doc)
    kept.sort(key=lambda item: float(item.score or 0.0), reverse=True)
    return kept, filtered
