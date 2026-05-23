from __future__ import annotations

from typing import Protocol

from src.domain.models import QueryRequestAnalysisResult, VisRAGRuleDocument


class RuleRetriever(Protocol):
    name: str

    def rank(
            self,
            documents: list[VisRAGRuleDocument],
            *,
            query: str,
            query_analysis: QueryRequestAnalysisResult,
            corpus_key: str | None = None,
    ) -> list[VisRAGRuleDocument]:
        ...


def document_text(document: VisRAGRuleDocument) -> str:
    metadata = document.metadata or {}
    metadata_terms = " ".join(str(value) for key, value in metadata.items() if key in {"chart_family", "task", "source", "record_type"})
    return " ".join([document.title, document.retrieval_text, document.prompt_text, metadata_terms])


def apply_rule_metadata_boost(
        score: float,
        document: VisRAGRuleDocument,
        query_analysis: QueryRequestAnalysisResult,
) -> float:
    metadata = document.metadata or {}
    boosted = float(score)
    if metadata.get("chart_family") and metadata.get("chart_family") == query_analysis.recommended_chart_family:
        boosted += 0.4
    if metadata.get("task") and metadata.get("task") == query_analysis.analysis_task:
        boosted += 0.4
    try:
        source_weight = float(metadata.get("source_weight", 1.0))
    except (TypeError, ValueError):
        source_weight = 1.0
    return boosted * max(0.1, min(source_weight, 3.0))
