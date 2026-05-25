from __future__ import annotations

import math

from src.domain.models import QueryRequestAnalysisResult, VisRAGRuleDocument
from src.visrag_core.rule_retrieval.base import apply_rule_metadata_boost, document_text


class LangChainEmbeddingRuleRetriever:
    name = "langchain_embeddings"

    def __init__(
            self,
            *,
            provider: str | None = None,
            model: str | None = None,
            base_url: str | None = None,
    ) -> None:
        self.provider = (provider or "ollama").strip().lower()
        self.model = model
        self.base_url = base_url
        self._embeddings = None
        self._doc_embeddings_by_key: dict[str, dict[str, list[float]]] = {}

    def rank(
            self,
            documents: list[VisRAGRuleDocument],
            *,
            query: str,
            query_analysis: QueryRequestAnalysisResult,
            corpus_key: str | None = None,
    ) -> list[VisRAGRuleDocument]:
        if not query.strip() or not documents:
            return []
        embeddings = self._get_embeddings()
        doc_embeddings = self._index(documents, corpus_key, embeddings)
        query_embedding = list(embeddings.embed_query(query))
        ranked: list[VisRAGRuleDocument] = []
        for document in documents:
            score = _cosine(query_embedding, doc_embeddings.get(document.doc_id) or [])
            score = apply_rule_metadata_boost(score, document, query_analysis)
            if score <= 0:
                continue
            ranked.append(document.model_copy(update={"score": round(score, 6)}))
        return sorted(ranked, key=lambda item: (-item.score, item.record_type, item.doc_id))

    def _index(self, documents: list[VisRAGRuleDocument], corpus_key: str | None, embeddings) -> dict[str, list[float]]:
        key = corpus_key or f"memory:{len(documents)}:{sum(hash(document.doc_id) for document in documents)}"
        cached = self._doc_embeddings_by_key.get(key)
        if cached is not None:
            return cached
        texts = [document_text(document) for document in documents]
        vectors = embeddings.embed_documents(texts)
        doc_embeddings = {document.doc_id: list(vector) for document, vector in zip(documents, vectors)}
        self._doc_embeddings_by_key[key] = doc_embeddings
        return doc_embeddings

    def _get_embeddings(self):
        if self._embeddings is not None:
            return self._embeddings
        provider = self.provider
        if provider == "hf":
            provider = "huggingface"
        if provider in {"langchain", "embedding", "embeddings"}:
            provider = "ollama"
        if provider == "ollama":
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError as exc:
                raise RuntimeError("Ollama embeddings require langchain-ollama.") from exc
            kwargs: dict[str, object] = {"model": self.model or "nomic-embed-text"}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._embeddings = OllamaEmbeddings(**kwargs)
            return self._embeddings
        if provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError as exc:
                raise RuntimeError("OpenAI embeddings require langchain-openai.") from exc
            kwargs: dict[str, object] = {"model": self.model or "text-embedding-3-small"}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._embeddings = OpenAIEmbeddings(**kwargs)
            return self._embeddings
        if provider == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError as exc:
                raise RuntimeError("HuggingFace embeddings require langchain-huggingface.") from exc
            self._embeddings = HuggingFaceEmbeddings(model_name=self.model or "sentence-transformers/all-MiniLM-L6-v2")
            return self._embeddings
        raise ValueError(f"Unsupported VisRAG embedding provider: {self.provider!r}")


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
