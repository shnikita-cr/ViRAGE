from __future__ import annotations

from dataclasses import dataclass

from src.visrag_core.rule_retrieval.base import RuleRetriever
from src.visrag_core.rule_retrieval.bm25 import BM25RuleRetriever
from src.visrag_core.rule_retrieval.embeddings import LangChainEmbeddingRuleRetriever
from src.visrag_core.rule_retrieval.keyword import KeywordRuleRetriever
from src.visrag_core.rule_retrieval.tfidf import TfidfRuleRetriever


@dataclass(frozen=True)
class RuleRetrieverOptions:
    backend: str = "bm25"
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_base_url: str | None = None

    def cache_key(self) -> str:
        return "|".join([
            self.backend.strip().lower(),
            str(self.embedding_provider or ""),
            str(self.embedding_model or ""),
            str(self.embedding_base_url or ""),
        ])


def build_rule_retriever(options: RuleRetrieverOptions) -> RuleRetriever:
    backend = (options.backend or "bm25").strip().lower()
    if backend == "keyword":
        return KeywordRuleRetriever()
    if backend == "bm25":
        return BM25RuleRetriever()
    if backend in {"tfidf", "tf-idf"}:
        return TfidfRuleRetriever()
    if backend in {"langchain", "embedding", "embeddings", "ollama", "openai", "huggingface", "hf"}:
        provider = options.embedding_provider
        if backend in {"ollama", "openai", "huggingface", "hf"}:
            provider = "huggingface" if backend == "hf" else backend
        return LangChainEmbeddingRuleRetriever(
            provider=provider,
            model=options.embedding_model,
            base_url=options.embedding_base_url,
        )
    raise ValueError(f"Unsupported VisRAG retriever backend: {options.backend!r}")
