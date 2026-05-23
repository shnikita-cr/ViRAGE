from __future__ import annotations

from collections import Counter

from src.domain.models import QueryRequestAnalysisResult, VisRAGRuleDocument
from src.visrag_core.rule_retrieval.base import apply_rule_metadata_boost, document_text
from src.visrag_core.text import tokens


class KeywordRuleRetriever:
    """Deterministic keyword-overlap retriever for runtime rule documents."""

    name = "keyword"

    def rank(
            self,
            documents: list[VisRAGRuleDocument],
            *,
            query: str,
            query_analysis: QueryRequestAnalysisResult,
            corpus_key: str | None = None,
    ) -> list[VisRAGRuleDocument]:
        query_tokens = set(tokens(query))
        if not query_tokens:
            return []
        ranked: list[VisRAGRuleDocument] = []
        for document in documents:
            doc_tokens = set(tokens(document_text(document)))
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens)
            if overlap <= 0:
                continue
            score = apply_rule_metadata_boost(overlap / max(len(query_tokens), 1), document, query_analysis)
            ranked.append(document.model_copy(update={"score": round(score, 6)}))
        return sorted(ranked, key=lambda item: (-item.score, item.record_type, item.doc_id))
