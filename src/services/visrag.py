from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.models import DataProfile, QueryRequestAnalysisResult, VisRAGGuidanceChunk
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.visrag_core import VisRAGCoreOptions, VisRAGEngine, create_visrag_store
from src.visrag_core.stores import VisRAGStore


@dataclass
class _CachedChunkCorpus:
    signature: dict[str, object]
    chunks: list[VisRAGGuidanceChunk]
    embeddings: dict[str, list[float]]


class VisRAGService(BaseService):
    """Runtime VisRAG over pre-embedded guidance chunks."""

    _cache: dict[str, _CachedChunkCorpus] = {}

    def invoke(
            self,
            query_analysis: QueryRequestAnalysisResult,
            data_profile: DataProfile,
            runtime: RuntimeContext,
            task_context: dict[str, Any] | None = None,
    ):
        opts = runtime.settings.visrag_runtime_options()
        store = create_visrag_store(
            backend=str(opts["store_backend"]),
            uri=opts["corpus_root"],
        )
        options = VisRAGCoreOptions(
            enabled=bool(opts["enabled"]),
            store_backend=str(opts["store_backend"]),
            top_k_chunks=int(opts["top_k_chunks"]),
            embedding_provider=opts["embedding_provider"],
            embedding_model=opts["embedding_model"],
            embedding_base_url=opts["embedding_base_url"],
        )
        signature = store.corpus_signature()
        cached = self._load(store, signature)
        return VisRAGEngine(
            store=store,
            options=options,
            chunks=cached.chunks,
            embeddings=cached.embeddings,
            corpus_signature=signature,
            reasoning_llm=runtime.reasoning_llm,
        ).invoke(query_analysis, data_profile, task_context=task_context)

    @classmethod
    def _load(cls, store: VisRAGStore, signature: dict[str, object]) -> _CachedChunkCorpus:
        key = str(signature.get("cache_key") or signature.get("hash") or store.corpus_uri or "unknown")
        cached = cls._cache.get(key)
        if cached is not None:
            return cached
        chunks = store.load_chunks()
        embeddings = store.load_embeddings()
        result = _CachedChunkCorpus(signature=signature, chunks=chunks, embeddings=embeddings)
        cls._cache.clear()
        cls._cache[key] = result
        return result
