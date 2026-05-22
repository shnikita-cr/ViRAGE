from __future__ import annotations

from src.domain.models import DataProfile, QueryRequestAnalysisResult


def build_typed_queries(query_analysis: QueryRequestAnalysisResult, data_profile: DataProfile) -> dict[str, str]:
    field_text = " ".join(query_analysis.selected_fields)
    bindings_text = " ".join(
        f"{slot} {binding.field} {binding.role}" for slot, binding in query_analysis.field_bindings.items()
    )
    aggregation = query_analysis.aggregation_plan or {}
    agg_text = " ".join(str(value) for value in aggregation.values()) if isinstance(aggregation, dict) else str(aggregation)
    variant_text = " ".join(
        variant.text for variant in query_analysis.query_variants
        if variant.kind in {"canonical", "chart_pattern_retrieval", "analysis_rule_retrieval", "repair_rule_retrieval"}
    )
    base = " ".join([
        query_analysis.normalized_query,
        query_analysis.analysis_task,
        query_analysis.recommended_chart_family,
        field_text,
        bindings_text,
        agg_text,
        variant_text,
    ]).strip()
    visible = " ".join(query_analysis.visual_judge_requirements.get("must_be_visible", []))
    failures = " ".join(query_analysis.visual_judge_requirements.get("critical_failures", []))
    domain_terms = " ".join(column.name for column in data_profile.columns)
    return {
        "chart_pattern": base,
        "readability_rule": f"{base} readable axis legend labels category grouping",
        "scale_plot_area_rule": f"{base} scale domain outlier plot area compressed empty space zero baseline",
        "vlm_readability_rule": f"{base} static png visible required fields tooltip not enough {visible} {failures}",
        "domain_semantics_rule": f"{query_analysis.normalized_query} {field_text} {domain_terms}",
    }
