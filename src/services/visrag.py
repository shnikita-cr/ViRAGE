from __future__ import annotations

from dataclasses import dataclass

from src.domain.models import DataProfile, QueryRequestAnalysisResult, VisRAGResult, VisRAGRuleDocument
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.visrag_core import VisRAGCoreOptions, VisRAGEngine, create_rule_corpus_repository
from src.visrag_core.constants import DEFAULT_TOP_K
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
        repository = create_rule_corpus_repository(
            backend=str(getattr(runtime.settings, "visrag_runtime_store_backend", "jsonl") or "jsonl"),
            uri=getattr(runtime.settings, "visrag_corpus_root", None),
        )
        options = VisRAGCoreOptions(
            enabled=bool(getattr(runtime.settings, "visrag_enabled", True)),
            retriever_name=str(getattr(runtime.settings, "visrag_retriever_backend", "bm25") or "bm25"),
            embedding_provider=getattr(runtime.settings, "visrag_embedding_provider", None),
            embedding_model=getattr(runtime.settings, "visrag_embedding_model", None),
            embedding_base_url=getattr(runtime.settings, "visrag_embedding_base_url", None),
            top_k_by_type=self._top_k_by_type(runtime),
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

    @staticmethod
    def _top_k_by_type(runtime: RuntimeContext) -> dict[str, int]:
        values: dict[str, int] = {}
        for record_type, default_value in DEFAULT_TOP_K.items():
            setting_name = f"visrag_top_k_{record_type}s"
            value = getattr(runtime.settings, setting_name, None)
            values[record_type] = max(0, int(value if value is not None else default_value))
        return values
