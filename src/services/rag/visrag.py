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


class VisRAGService(BaseService):
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
            backend=str(opts["corpus_source"]),
            uri=opts["corpus_root"],
        )
        options = VisRAGCoreOptions(
            enabled=bool(opts["enabled"]),
            corpus_source=str(opts["corpus_source"]),
            vector_index=str(opts["vector_index"]),
            retrieval_backend=str(opts["retrieval_backend"]),
            top_k_chunks=int(opts["top_k_chunks"]),
            hybrid_method=str(opts["hybrid_method"]),
            hybrid_weight=float(opts["hybrid_weight"]),
            hybrid_rrf_k=float(opts["hybrid_rrf_k"]),
            candidate_pool_size=int(opts["candidate_pool_size"]),
            metadata_weight_manual_feedback=float(opts["metadata_weight_manual_feedback"]),
            metadata_weight_scientific_figure=float(opts["metadata_weight_scientific_figure"]),
            metadata_weight_min=float(opts["metadata_weight_min"]),
            metadata_weight_max=float(opts["metadata_weight_max"]),
            embedding_provider=opts["embedding_provider"],
            embedding_model=opts["embedding_model"],
            embedding_base_url=opts["embedding_base_url"],
            chroma_persist_dir=opts["chroma_persist_dir"],
            chroma_collection_name=str(opts["chroma_collection_name"]),
        )
        signature = store.corpus_signature()
        cached = self._load(store, signature)
        return VisRAGEngine(
            store=store,
            options=options,
            chunks=cached.chunks,
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
        result = _CachedChunkCorpus(signature=signature, chunks=chunks)
        cls._cache.clear()
        cls._cache[key] = result
        return result
