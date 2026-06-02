from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from src.domain.models import QueryRequestAnalysisResult, VisRAGRuleDocument
from src.visrag_core.text import tokens


@dataclass(frozen=True)
class RuleRetrieverOptions:
    backend: str = "bm25"
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_base_url: str | None = None


class _LexicalRuleRetriever:
    def __init__(self, options: RuleRetrieverOptions) -> None:
        self.options = options

    def rank(
            self,
            documents: list[VisRAGRuleDocument],
            *,
            query: str,
            query_analysis: QueryRequestAnalysisResult | None = None,
            corpus_key: str | None = None,
    ) -> list[VisRAGRuleDocument]:
        backend = (self.options.backend or "bm25").lower()
        query_tokens = tokens(query)
        if query_analysis is not None:
            query_tokens.extend(tokens(" ".join([
                query_analysis.analysis_task,
                " ".join(query_analysis.selected_fields),
            ])))
        if not query_tokens:
            return list(documents)
        if backend == "keyword":
            return self._rank_keyword(documents, query_tokens)
        if backend == "tfidf":
            return self._rank_tfidf(documents, query_tokens)
        return self._rank_bm25(documents, query_tokens)

    @staticmethod
    def _doc_text(document: VisRAGRuleDocument) -> str:
        return " ".join([document.title, document.retrieval_text, document.prompt_text])

    def _rank_keyword(self, documents: list[VisRAGRuleDocument], query_tokens: list[str]) -> list[VisRAGRuleDocument]:
        query_set = set(query_tokens)
        ranked = []
        for document in documents:
            doc_tokens = tokens(self._doc_text(document))
            overlap = len(query_set & set(doc_tokens))
            score = float(overlap) + float(document.score or 0.0)
            ranked.append(document.model_copy(update={"score": round(score, 6)}))
        return sorted(ranked, key=lambda item: (-item.score, item.doc_id))

    def _rank_tfidf(self, documents: list[VisRAGRuleDocument], query_tokens: list[str]) -> list[VisRAGRuleDocument]:
        tokenized = [tokens(self._doc_text(document)) for document in documents]
        n_docs = max(1, len(documents))
        df: dict[str, int] = {}
        for doc_tokens in tokenized:
            for token in set(doc_tokens):
                df[token] = df.get(token, 0) + 1
        query_set = set(query_tokens)
        ranked = []
        for document, doc_tokens in zip(documents, tokenized, strict=False):
            counts: dict[str, int] = {}
            for token in doc_tokens:
                counts[token] = counts.get(token, 0) + 1
            score = 0.0
            for token in query_set:
                tf = counts.get(token, 0)
                if tf:
                    score += (1.0 + math.log(tf)) * math.log((n_docs + 1) / (df.get(token, 0) + 1))
            ranked.append(document.model_copy(update={"score": round(score, 6)}))
        return sorted(ranked, key=lambda item: (-item.score, item.doc_id))

    def _rank_bm25(self, documents: list[VisRAGRuleDocument], query_tokens: list[str]) -> list[VisRAGRuleDocument]:
        tokenized = [tokens(self._doc_text(document)) for document in documents]
        n_docs = max(1, len(documents))
        avg_len = sum(len(item) for item in tokenized) / n_docs if tokenized else 1.0
        df: dict[str, int] = {}
        for doc_tokens in tokenized:
            for token in set(doc_tokens):
                df[token] = df.get(token, 0) + 1
        k1 = 1.5
        b = 0.75
        ranked = []
        for document, doc_tokens in zip(documents, tokenized, strict=False):
            counts: dict[str, int] = {}
            for token in doc_tokens:
                counts[token] = counts.get(token, 0) + 1
            doc_len = max(1, len(doc_tokens))
            score = 0.0
            for token in set(query_tokens):
                tf = counts.get(token, 0)
                if not tf:
                    continue
                idf = math.log(1.0 + (n_docs - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5))
                denom = tf + k1 * (1.0 - b + b * doc_len / max(avg_len, 1.0))
                score += idf * (tf * (k1 + 1.0)) / denom
            ranked.append(document.model_copy(update={"score": round(score, 6)}))
        return sorted(ranked, key=lambda item: (-item.score, item.doc_id))


def build_rule_retriever(options: RuleRetrieverOptions) -> _LexicalRuleRetriever:
    return _LexicalRuleRetriever(options)
