from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from src.domain.models import QueryRequestAnalysisResult, VisRAGRuleDocument
from src.visrag_core.rule_retrieval.base import apply_rule_metadata_boost, document_text
from src.visrag_core.text import tokens


@dataclass(frozen=True)
class _BM25Index:
    doc_tokens: dict[str, list[str]]
    doc_freq: dict[str, int]
    avg_len: float
    doc_count: int


class BM25RuleRetriever:
    """Small dependency-free BM25 retriever for runtime rule documents."""

    name = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._indexes: dict[str, _BM25Index] = {}

    def rank(
            self,
            documents: list[VisRAGRuleDocument],
            *,
            query: str,
            query_analysis: QueryRequestAnalysisResult,
            corpus_key: str | None = None,
    ) -> list[VisRAGRuleDocument]:
        query_tokens = tokens(query)
        if not query_tokens:
            return []
        index = self._index(documents, corpus_key)
        query_counts = Counter(query_tokens)
        ranked: list[VisRAGRuleDocument] = []
        for document in documents:
            doc_tokens = index.doc_tokens.get(document.doc_id) or []
            if not doc_tokens:
                continue
            score = self._score_document(query_counts, doc_tokens, index)
            score = apply_rule_metadata_boost(score, document, query_analysis)
            if score <= 0:
                continue
            ranked.append(document.model_copy(update={"score": round(score, 6)}))
        return sorted(ranked, key=lambda item: (-item.score, item.record_type, item.doc_id))

    def _index(self, documents: list[VisRAGRuleDocument], corpus_key: str | None) -> _BM25Index:
        key = corpus_key or f"memory:{len(documents)}:{sum(hash(document.doc_id) for document in documents)}"
        cached = self._indexes.get(key)
        if cached is not None:
            return cached
        doc_tokens: dict[str, list[str]] = {}
        doc_freq: dict[str, int] = defaultdict(int)
        total_len = 0
        for document in documents:
            values = tokens(document_text(document))
            doc_tokens[document.doc_id] = values
            total_len += len(values)
            for token in set(values):
                doc_freq[token] += 1
        doc_count = max(len(documents), 1)
        index = _BM25Index(
            doc_tokens=doc_tokens,
            doc_freq=dict(doc_freq),
            doc_count=doc_count,
            avg_len=max(total_len / doc_count, 1.0),
        )
        self._indexes[key] = index
        return index

    def _score_document(self, query_counts: Counter[str], doc_tokens: list[str], index: _BM25Index) -> float:
        doc_counts = Counter(doc_tokens)
        doc_len = max(len(doc_tokens), 1)
        score = 0.0
        for token, query_count in query_counts.items():
            tf = doc_counts.get(token, 0)
            if tf <= 0:
                continue
            df = index.doc_freq.get(token, 0)
            idf = math.log(1.0 + (index.doc_count - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / index.avg_len)
            score += idf * (tf * (self.k1 + 1.0) / denom) * max(query_count, 1)
        return score
