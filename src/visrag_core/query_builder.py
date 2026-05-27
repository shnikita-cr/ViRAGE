from __future__ import annotations

from src.domain.models import DataProfile, QueryRequestAnalysisResult


def build_visrag_query(query_analysis: QueryRequestAnalysisResult, data_profile: DataProfile) -> str:
    fields = " ".join(query_analysis.selected_fields)
    bindings = " ".join(
        f"{slot} {binding.field} {binding.role}" for slot, binding in query_analysis.field_bindings.items()
    )
    variants = " ".join(variant.text for variant in query_analysis.query_variants[:6])
    columns = " ".join(column.name for column in data_profile.columns[:30])
    return " ".join([
        query_analysis.normalized_query,
        query_analysis.analysis_task,
        query_analysis.recommended_chart_family,
        fields,
        bindings,
        variants,
        columns,
    ]).strip()
