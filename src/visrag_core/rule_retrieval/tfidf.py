from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from src.domain.models import QueryRequestAnalysisResult, VisRAGRuleDocument
from src.visrag_core.rule_retrieval.base import apply_rule_metadata_boost, document_text
from src.visrag_core.text import tokens


@dataclass(frozen=True)
class _TfidfIndex:
    vectors: dict[str, dict[str, float]]
    idf: dict[str, float]


class TfidfRuleRetriever:
    name = "tfidf"

    def __init__(self) -> None:
        self._indexes: dict[str, _TfidfIndex] = {}

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
        query_vector = _vectorize(query_tokens, index.idf)
        ranked: list[VisRAGRuleDocument] = []
        for document in documents:
            score = _cosine(query_vector, index.vectors.get(document.doc_id) or {})
            score = apply_rule_metadata_boost(score, document, query_analysis)
            if score <= 0:
                continue
            ranked.append(document.model_copy(update={"score": round(score, 6)}))
        return sorted(ranked, key=lambda item: (-item.score, item.record_type, item.doc_id))

    def _index(self, documents: list[VisRAGRuleDocument], corpus_key: str | None) -> _TfidfIndex:
        key = corpus_key or f"memory:{len(documents)}:{sum(hash(document.doc_id) for document in documents)}"
        cached = self._indexes.get(key)
        if cached is not None:
            return cached
        tokenized = {document.doc_id: tokens(document_text(document)) for document in documents}
        doc_freq: dict[str, int] = defaultdict(int)
        for values in tokenized.values():
            for token in set(values):
                doc_freq[token] += 1
        doc_count = max(len(documents), 1)
        idf = {token: math.log((doc_count + 1) / (count + 1)) + 1.0 for token, count in doc_freq.items()}
        index = _TfidfIndex(
            vectors={doc_id: _vectorize(values, idf) for doc_id, values in tokenized.items()},
            idf=idf,
        )
        self._indexes[key] = index
        return index


def _vectorize(values: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(values)
    total = max(sum(counts.values()), 1)
    return {token: (count / total) * idf.get(token, 1.0) for token, count in counts.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
