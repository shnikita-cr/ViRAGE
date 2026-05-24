from __future__ import annotations

from dataclasses import dataclass

from src.domain.models import DataProfile, QueryRequestAnalysisResult, VisRAGResult, VisRAGRuleDocument
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.visrag_core import VisRAGCoreOptions, VisRAGEngine, create_rule_corpus_repository
from src.visrag_core.rule_retrieval import RuleRetriever, build_rule_retriever
from src.visrag_core.stores import RuleCorpusRepository


@dataclass
class _CachedCorpus:
    signature: dict[str, object]
    documents: list[VisRAGRuleDocument]


class VisRAGService(BaseService):
    """Application service wrapper around runtime rule-guidance VisRAG."""

    _corpus_cache: dict[str, _CachedCorpus] = {}
    _retriever_cache: dict[str, RuleRetriever] = {}

    def invoke(
            self,
            query_analysis: QueryRequestAnalysisResult,
            data_profile: DataProfile,
            runtime: RuntimeContext,
    ) -> VisRAGResult:
        visrag_options = runtime.settings.visrag_runtime_options()
        repository = create_rule_corpus_repository(
            backend=str(visrag_options["store_backend"] or "jsonl"),
            uri=visrag_options["corpus_root"],
        )
        options = VisRAGCoreOptions(
            enabled=bool(visrag_options["enabled"]),
            retriever_name=str(visrag_options["retriever_backend"] or "bm25"),
            embedding_provider=visrag_options["embedding_provider"],
            embedding_model=visrag_options["embedding_model"],
            embedding_base_url=visrag_options["embedding_base_url"],
            top_k_by_type=dict(visrag_options["top_k_by_type"]),
        )
        signature = repository.corpus_signature()
        documents = [] if not options.enabled else self._load_documents(repository, signature)
        retriever = self._retriever(options, signature)
        return VisRAGEngine(
            repository=repository,
            options=options,
            documents=documents,
            retriever=retriever,
            corpus_signature=signature | {"document_count": len(documents)},
        ).invoke(query_analysis, data_profile)

    @classmethod
    def _load_documents(
            cls,
            repository: RuleCorpusRepository,
            signature: dict[str, object],
    ) -> list[VisRAGRuleDocument]:
        cache_key = str(signature.get("cache_key") or signature.get("hash") or repository.corpus_uri or "unknown")
        cached = cls._corpus_cache.get(cache_key)
        if cached is not None:
            return cached.documents
        documents = repository.load_documents()
        cls._corpus_cache.clear()
        cls._corpus_cache[cache_key] = _CachedCorpus(signature=signature, documents=documents)
        return documents

    @classmethod
    def _retriever(cls, options: VisRAGCoreOptions, signature: dict[str, object]) -> RuleRetriever:
        retriever_options = options.retriever_options()
        cache_key = f"{retriever_options.cache_key()}|{signature.get('hash') or signature.get('cache_key') or ''}"
        cached = cls._retriever_cache.get(cache_key)
        if cached is not None:
            return cached
        retriever = build_rule_retriever(retriever_options)
        cls._retriever_cache.clear()
        cls._retriever_cache[cache_key] = retriever
        return retriever

