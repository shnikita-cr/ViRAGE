from __future__ import annotations

from src.domain.models import DataProfile, QueryRequestAnalysisResult, VisRAGRuleDocument
from src.visrag_core.chart_types import canonicalize_chart_type

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
    recommended = canonicalize_chart_type(query_analysis.recommended_chart_family)
    has_geo = _profile_has_geo(data_profile)
    kept: list[VisRAGRuleDocument] = []
    filtered: list[dict[str, str]] = []
    for document in documents:
        family = _doc_chart_family(document)
        is_geo = "choropleth" in family or "geo" in family or "map" in family
        if is_geo and not has_geo and recommended not in {"geoshape"}:
            filtered.append({"doc_id": document.doc_id, "reason": "incompatible_chart_family:no_geo_fields"})
            continue
        boost = 1.0
        if recommended and recommended != "unknown":
            if recommended in family or family in recommended:
                boost = 1.25
            elif recommended == "line" and "line" in family:
                boost = 1.25
        kept.append(document.model_copy(update={"score": round(float(document.score or 0.0) * boost, 6)}))
    return sorted(kept, key=lambda item: (-item.score, item.doc_id)), filtered
