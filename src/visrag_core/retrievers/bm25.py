from __future__ import annotations

import math
from collections import Counter

from src.domain.models import QueryRequestAnalysisResult, VisRAGRuleDocument
from src.visrag_core.text import tokens


class BM25LikeRuleRetriever:
    """Small dependency-free lexical retriever for runtime rule documents.

    AutoRAG remains responsible for offline retriever optimization. This class is
    a lightweight runtime fallback and a stable implementation for local JSONL
    rule stores.
    """

    name = "bm25"

    def rank(
            self,
            documents: list[VisRAGRuleDocument],
            *,
            query: str,
            query_analysis: QueryRequestAnalysisResult,
    ) -> list[VisRAGRuleDocument]:
        query_tokens = tokens(query)
        query_counts = Counter(query_tokens)
        if not query_counts:
            return []
        ranked: list[VisRAGRuleDocument] = []
        for doc in documents:
            doc_tokens = tokens(" ".join([doc.title, doc.retrieval_text, doc.prompt_text]))
            if not doc_tokens:
                continue
            doc_counts = Counter(doc_tokens)
            overlap = sum(min(count, doc_counts.get(token, 0)) for token, count in query_counts.items())
            norm = math.sqrt(sum(value * value for value in doc_counts.values())) or 1.0
            score = overlap / norm
            metadata = doc.metadata or {}
            if metadata.get("chart_family") and metadata.get("chart_family") == query_analysis.recommended_chart_family:
                score += 0.4
            if metadata.get("task") and metadata.get("task") == query_analysis.analysis_task:
                score += 0.4
            if score <= 0:
                continue
            ranked.append(doc.model_copy(update={"score": round(score, 6)}))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked
