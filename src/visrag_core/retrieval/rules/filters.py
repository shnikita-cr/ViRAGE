from __future__ import annotations

from src.domain.models import DataProfile, QueryRequestAnalysisResult, VisRAGRuleDocument

_GEO_HINTS = {"country", "state", "region_code", "latitude", "longitude", "lat", "lon", "geo", "map"}


def _profile_has_geo(profile: DataProfile) -> bool:
    for column in profile.columns:
        text = " ".join([column.name, str(column.role or ""), str(column.dtype or "")]).lower()
        if any(hint in text for hint in _GEO_HINTS):
            return True
    return False


def _doc_chart_family(document: VisRAGRuleDocument) -> str:
    metadata = document.metadata or {}
    return str(metadata.get("chart_family") or metadata.get("chart_type") or document.title).strip().lower()


def rerank_by_compatibility(
        documents: list[VisRAGRuleDocument],
        query_analysis: QueryRequestAnalysisResult,
        data_profile: DataProfile,
) -> tuple[list[VisRAGRuleDocument], list[dict[str, str]]]:
    """Remove clearly incompatible documents without using chart-family recommendations.

    Query analysis no longer recommends chart families. This compatibility pass is
    therefore limited to data-schema constraints that are safe to infer, such as
    rejecting map/choropleth guidance when no geographic fields are present.
    """
    has_geo = _profile_has_geo(data_profile)
    kept: list[VisRAGRuleDocument] = []
    filtered: list[dict[str, str]] = []
    for document in documents:
        family = _doc_chart_family(document)
        is_geo = "choropleth" in family or "geo" in family or "map" in family
        if is_geo and not has_geo:
            filtered.append({"doc_id": document.doc_id, "reason": "incompatible_chart_family:no_geo_fields"})
            continue
        kept.append(document)
    return sorted(kept, key=lambda item: (-item.score, item.doc_id)), filtered
