from __future__ import annotations

from collections.abc import Iterable

from src.domain.models import QueryRequestAnalysisResult, VisRAGGenerationGuidance, VisRAGRuleDocument
from src.visrag_core.text import unique_text


def compose_generation_guidance(
        selected_by_type: dict[str, list[VisRAGRuleDocument]],
        query_analysis: QueryRequestAnalysisResult,
) -> VisRAGGenerationGuidance:
    chart_patterns = selected_by_type.get("chart_pattern", [])
    readability = selected_by_type.get("readability_rule", [])
    scale_rules = selected_by_type.get("scale_plot_area_rule", [])
    vlm_rules = selected_by_type.get("vlm_readability_rule", [])
    domain_rules = selected_by_type.get("domain_semantics_rule", [])

    bullets: list[str] = []
    bullets.append(
        "Query analysis is the source of truth for task, selected fields, chart family, and aggregation. "
        "Use retrieved rules only as supporting guidance."
    )
    bullets.append(
        f"Target chart family: {query_analysis.recommended_chart_family}; task: {query_analysis.analysis_task}."
    )
    for doc in [*chart_patterns, *readability, *scale_rules, *vlm_rules, *domain_rules]:
        if doc.prompt_text and doc.prompt_text not in bullets:
            bullets.append(doc.prompt_text)
    if not scale_rules:
        bullets.append(
            "Plot-area policy: for line/scatter charts, avoid forcing zero when it compresses the main pattern; "
            "for bar charts, preserve a truthful zero baseline by default."
        )
    if not vlm_rules:
        bullets.append(
            "VLM readability: required fields must be visible in the static PNG; tooltip-only encoding is not enough."
        )
    prompt_text = "VisRAG rule guidance:\n" + "\n".join(f"- {item}" for item in unique_text(bullets)[:12])
    return VisRAGGenerationGuidance(
        chart_patterns=chart_patterns,
        readability_rules=readability,
        scale_plot_area_rules=scale_rules,
        vlm_readability_rules=vlm_rules,
        domain_semantics_rules=domain_rules,
        anti_patterns=collect_avoidance_rules([*chart_patterns, *readability, *scale_rules, *vlm_rules, *domain_rules]),
        repair_hints=[],
        prompt_text=prompt_text,
    )


def collect_avoidance_rules(docs: Iterable[VisRAGRuleDocument]) -> list[str]:
    values: list[str] = []
    for doc in docs:
        avoid = doc.metadata.get("avoid") if isinstance(doc.metadata, dict) else None
        if isinstance(avoid, list):
            values.extend(str(item) for item in avoid if str(item).strip())
    return unique_text(values)[:8]
