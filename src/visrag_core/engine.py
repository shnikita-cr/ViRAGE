from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.domain.models import (
    DataProfile,
    QueryRequestAnalysisResult,
    VisRAGDebugRetrieval,
    VisRAGDiagnostics,
    VisRAGGenerationGuidance,
    VisRAGGuidanceChunk,
    VisRAGRuleDocument,
    VisRAGResult,
    VisRAGRetrievedChunk,
)
from src.llm.helpers import invoke_structured
from src.visrag_core.chroma_index import ChromaChunkRetriever, validate_chroma_manifest
from src.visrag_core.hybrid_retrieval_scorer import HybridRetrievalScorer
from src.visrag_core.lexical_retriever import RankBM25ChunkRetriever
from src.visrag_core.metadata_weighting import MetadataWeightingPolicy
from src.visrag_core.query_builder import build_visrag_query
from src.visrag_core.stores import JsonlVisRAGStore, VisRAGStore
from src.visrag_core.task_context import task_context_prompt_block


@dataclass(frozen=True)
class VisRAGCoreOptions:
    enabled: bool = True
    corpus_source: str = "jsonl"
    vector_index: str = "chroma"
    retrieval_backend: str = "hybrid"
    top_k_chunks: int = 8
    hybrid_method: str = "cc"
    hybrid_weight: float = 0.1
    hybrid_rrf_k: float = 60.0
    candidate_pool_size: int = 64
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    metadata_weight_manual_feedback: float = 1.5
    metadata_weight_scientific_figure: float = 1.2
    metadata_weight_min: float = 0.1
    metadata_weight_max: float = 4.0
    chroma_persist_dir: str | Path = "resources/chroma/virage_guidance_chunks_nomic_embed_text_latest"
    chroma_collection_name: str = "virage_guidance_chunks_nomic_embed_text_latest"


class _VisRAGResponseSchema(BaseModel):
    applicable_rules: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)
    feedback_warnings: list[str] = Field(default_factory=list)


class VisRAGEngine:
    def __init__(
            self,
            *,
            store: VisRAGStore,
            options: VisRAGCoreOptions,
            corpus_signature: dict[str, Any] | None = None,
            chunks: list[VisRAGGuidanceChunk] | None = None,
            reasoning_llm: Any | None = None,
    ) -> None:
        self.store = store
        self.options = options
        self._signature = corpus_signature or {}
        self._chunks = chunks
        self.reasoning_llm = reasoning_llm

    def invoke(
            self,
            query_analysis: QueryRequestAnalysisResult,
            data_profile: DataProfile,
            task_context: dict[str, Any] | None = None,
    ) -> VisRAGResult:
        if not self.options.enabled:
            raise RuntimeError("VisRAG is disabled. Enable visrag_enabled=true or remove the visrag pipeline node.")
        chunks = self._chunks if self._chunks is not None else self.store.load_chunks()
        query = build_visrag_query(query_analysis, data_profile, task_context=task_context)
        retrieved = self._retrieve(query, query_analysis, chunks)
        guidance = self._generate_response(query_analysis, data_profile, retrieved, task_context=task_context)
        diagnostics = VisRAGDiagnostics(
            retrieved_count=len(retrieved),
            retrieved_count_by_type=self._count_by_kind(retrieved),
            corpus_backend=self._retrieval_runtime_backend(),
            corpus_uri=self.store.corpus_uri,
            corpus_hash=str(self._signature.get("hash") or ""),
        )
        debug = VisRAGDebugRetrieval(
            retrieval_query=query,
            retrieved_chunks=retrieved,
            retrieved_documents=retrieved,
            scores=[{"chunk_id": chunk.chunk_id, "score": chunk.score, "source_id": chunk.source_id} for chunk in retrieved],
            task_context=dict(task_context or {}),
        )
        return VisRAGResult(
            corpus_status=self._corpus_status(chunks),
            retrieval_strategy=self._retrieval_strategy(),
            generation_guidance=guidance,
            debug_retrieval=debug,
            diagnostics=diagnostics,
        )


    def _corpus_status(self, chunks: list[VisRAGGuidanceChunk]) -> dict[str, Any]:
        backend = self._normalized_retrieval_backend()
        status: dict[str, Any] = {
            "enabled": True,
            "chunks": len(chunks),
            "corpus_source": {
                "backend": "guidance_chunks",
                "uri": self.store.corpus_uri,
                "signature": self._signature,
            },
            "retrieval_mode": backend,
            "retrieval_backend": self._retrieval_runtime_backend(),
            "metadata_weighting": self._metadata_weighting_policy().diagnostics(),
        }
        if backend in {"semantic", "hybrid"}:
            status["vector_index"] = {
                "backend": "chroma",
                "persist_dir": str(self.options.chroma_persist_dir),
                "collection_name": self.options.chroma_collection_name,
                "embedding_provider": self.options.embedding_provider or "ollama",
                "embedding_model": self.options.embedding_model or "nomic-embed-text:latest",
            }
        if backend in {"lexical", "hybrid"}:
            status["lexical_index"] = {"backend": "rank_bm25"}
        return status

    def _retrieval_runtime_backend(self) -> str:
        backend = self._normalized_retrieval_backend()
        if backend == "semantic":
            return "chroma"
        if backend == "hybrid":
            return "chroma+rank_bm25"
        if backend == "lexical":
            return "rank_bm25"
        return backend

    def _retrieve(
            self,
            query: str,
            query_analysis: QueryRequestAnalysisResult,
            chunks: list[VisRAGGuidanceChunk],
    ) -> list[VisRAGRetrievedChunk]:
        if not query.strip():
            return []
        backend = self._normalized_retrieval_backend()
        lexical_query = self._lexical_query(query, query_analysis)
        if backend == "semantic":
            semantic_scores = self._semantic_scores(query, chunks)
            return self._rank_from_scores(chunks, semantic_scores, score_kind="semantic")
        if backend == "lexical":
            lexical_scores = self._lexical_scores(lexical_query, chunks)
            return self._rank_from_scores(chunks, lexical_scores, score_kind="lexical")
        if backend == "hybrid":
            semantic_scores = self._semantic_scores(query, chunks)
            lexical_scores = self._top_lexical_scores(lexical_query, chunks)
            hybrid_scores = self._hybrid_scores(semantic_scores, lexical_scores)
            return self._rank_from_scores(chunks, hybrid_scores, score_kind="hybrid")
        raise RuntimeError(
            f"Unsupported VisRAG retrieval backend: {self.options.retrieval_backend!r}. "
            "Use one of: semantic, hybrid, lexical."
        )

    def _semantic_scores(self, query: str, chunks: list[VisRAGGuidanceChunk]) -> dict[str, float]:
        self._validate_semantic_index(chunks)
        retriever = ChromaChunkRetriever(
            persist_dir=Path(self.options.chroma_persist_dir),
            collection_name=self.options.chroma_collection_name,
            embedding_provider=self.options.embedding_provider,
            embedding_model=self.options.embedding_model,
            embedding_base_url=self.options.embedding_base_url,
        )
        return retriever.score(query=query, chunks=chunks, limit=self._candidate_pool_limit())

    def _validate_semantic_index(self, chunks: list[VisRAGGuidanceChunk]) -> None:
        if not isinstance(self.store, JsonlVisRAGStore):
            raise RuntimeError("Semantic VisRAG retrieval requires JsonlVisRAGStore as corpus source.")
        validate_chroma_manifest(
            persist_dir=Path(self.options.chroma_persist_dir),
            chunks_path=self.store.chunks_path,
            collection_name=self.options.chroma_collection_name,
            embedding_provider=self.options.embedding_provider,
            embedding_model=self.options.embedding_model,
            expected_chunk_count=len(chunks),
        )

    @staticmethod
    def _lexical_query(query: str, query_analysis: QueryRequestAnalysisResult) -> str:
        return " ".join([
            query,
            query_analysis.analysis_task,
            " ".join(query_analysis.selected_fields),
        ]).strip()

    @staticmethod
    def _lexical_scores(query: str, chunks: list[VisRAGGuidanceChunk]) -> dict[str, float]:
        return RankBM25ChunkRetriever().score(query=query, chunks=chunks)

    def _top_lexical_scores(self, query: str, chunks: list[VisRAGGuidanceChunk]) -> dict[str, float]:
        scores = self._lexical_scores(query, chunks)
        ranked = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
        keep = {chunk_id for chunk_id, score in ranked[:self._candidate_pool_limit()] if float(score) > 0.0}
        return {chunk_id: score for chunk_id, score in scores.items() if chunk_id in keep}

    def _hybrid_scores(self, semantic_scores: dict[str, float], lexical_scores: dict[str, float]) -> dict[str, float]:
        return HybridRetrievalScorer(
            method=self.options.hybrid_method,
            semantic_weight=self.options.hybrid_weight,
            rrf_k=self.options.hybrid_rrf_k,
        ).score(semantic_scores, lexical_scores)

    def _rank_from_scores(
            self,
            chunks: list[VisRAGGuidanceChunk],
            scores: dict[str, float],
            *,
            score_kind: str,
    ) -> list[VisRAGRetrievedChunk]:
        raw_ranked = sorted(
            [chunk for chunk in chunks if float(scores.get(chunk.chunk_id, 0.0) or 0.0) > 0],
            key=lambda chunk: (-float(scores.get(chunk.chunk_id, 0.0) or 0.0), chunk.source_id, chunk.chunk_id),
        )
        ranked: list[VisRAGRetrievedChunk] = []
        for chunk in raw_ranked[:self._candidate_pool_limit()]:
            raw_score = float(scores.get(chunk.chunk_id, 0.0) or 0.0)
            final_score = raw_score * self._metadata_weight(chunk)
            metadata = dict(chunk.metadata or {})
            metadata["retrieval_score_kind"] = score_kind
            metadata["raw_retrieval_score"] = round(raw_score, 6)
            metadata["candidate_pool_size"] = self._candidate_pool_limit()
            ranked.append(
                VisRAGRetrievedChunk(
                    **chunk.model_dump(exclude={"score", "metadata"}),
                    metadata=metadata,
                    score=round(final_score, 6),
                )
            )
        limit = max(1, int(self.options.top_k_chunks))
        return sorted(ranked, key=lambda item: (-item.score, item.source_id, item.chunk_id))[:limit]

    def _candidate_pool_limit(self) -> int:
        return max(int(self.options.top_k_chunks), int(self.options.candidate_pool_size))

    def _normalized_retrieval_backend(self) -> str:
        return (self.options.retrieval_backend or "semantic").strip().lower()

    def _retrieval_strategy(self) -> str:
        backend = self._normalized_retrieval_backend()
        if backend == "hybrid":
            return (
                f"chunk_guidance:hybrid:chroma+rank_bm25:{self.options.hybrid_method}:"
                f"collection={self.options.chroma_collection_name}:"
                f"embedding={self.options.embedding_model}:weight={self.options.hybrid_weight}:top_k={self.options.top_k_chunks}"
            )
        if backend == "semantic":
            return (
                f"chunk_guidance:semantic:chroma:"
                f"collection={self.options.chroma_collection_name}:"
                f"embedding={self.options.embedding_model}:top_k={self.options.top_k_chunks}"
            )
        if backend == "lexical":
            return f"chunk_guidance:lexical:rank_bm25:top_k={self.options.top_k_chunks}"
        return f"chunk_guidance:{backend}"

    def _metadata_weight(self, chunk: VisRAGGuidanceChunk) -> float:
        return self._metadata_weighting_policy().weight(chunk)

    def _metadata_weighting_policy(self) -> MetadataWeightingPolicy:
        return MetadataWeightingPolicy(
            manual_feedback_weight=self.options.metadata_weight_manual_feedback,
            scientific_figure_weight=self.options.metadata_weight_scientific_figure,
            min_weight=self.options.metadata_weight_min,
            max_weight=self.options.metadata_weight_max,
        )

    def _generate_response(
            self,
            query_analysis: QueryRequestAnalysisResult,
            data_profile: DataProfile,
            chunks: list[VisRAGRetrievedChunk],
            task_context: dict[str, Any] | None = None,
    ) -> VisRAGGenerationGuidance:
        source_refs = [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "source_name": chunk.source_name,
                "title": chunk.title,
                "score": chunk.score,
                "source_path": chunk.source_path,
                "url": chunk.url,
            }
            for chunk in chunks
        ]
        categorized = self._categorized_rule_documents(chunks)
        if self.reasoning_llm is None:
            guidance = VisRAGGenerationGuidance(
                chart_patterns=categorized["chart_pattern"],
                readability_rules=categorized["readability_rule"],
                scale_plot_area_rules=categorized["scale_plot_area_rule"],
                vlm_readability_rules=categorized["vlm_readability_rule"],
                domain_semantics_rules=self._gate_domain_semantics(categorized["domain_semantics_rule"], query_analysis, data_profile),
                applicable_rules=[chunk.text for chunk in chunks if str(chunk.source_kind) not in {"domain_semantics_rule"}],
                quality_checks=[chunk.text for chunk in chunks if str(chunk.source_kind) in {"readability_rule", "vlm_readability_rule"}],
                source_refs=source_refs,
            )
            guidance.prompt_text = self._to_prompt_text(guidance, task_context=task_context)
            return guidance

        payload = {
            "query_analysis": {
                "normalized_query": query_analysis.normalized_query,
                "analysis_task": query_analysis.analysis_task,
                "selected_fields": query_analysis.selected_fields,
            },
            "selected_analytical_subtask": dict(task_context or {}),
            "data_profile": {
                "columns": [column.model_dump() for column in data_profile.columns[:30]],
                "row_count": getattr(data_profile, "row_count", None),
            },
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "source_kind": chunk.source_kind,
                    "title": chunk.title,
                    "text": chunk.text[:2400],
                }
                for chunk in chunks
            ],
        }
        stage = str((task_context or {}).get("stage") or "spec_generation").strip().lower()
        if stage == "planning":
            instruction = (
                "Generate VisRAG guidance for analysis planning. "
                "Use only the retrieved chunks and the given data/query context. "
                "Return practical constraints for selecting subtasks, fields, metric semantics, ranking strategy, scale strategy and visual constraints. "
                "Do not output Vega-Lite code or examples."
            )
        else:
            instruction = (
                "Generate final VisRAG guidance for Vega-Lite spec generation. "
                "Use only the retrieved chunks and the given data/query context. "
                "If selected_analytical_subtask is provided, treat it as fixed: do not replace it with another task. "
                "Do not output Vega-Lite code or examples. Return concrete practical guidance."
            )
        prompt = f"{instruction}\n\nContext JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
        parsed = invoke_structured(
            self.reasoning_llm,
            prompt,
            _VisRAGResponseSchema,
            stage="visrag_guidance_generation",
            role="reasoning",
            max_attempts=2,
        )
        guidance = VisRAGGenerationGuidance(
            chart_patterns=categorized["chart_pattern"],
            readability_rules=categorized["readability_rule"],
            scale_plot_area_rules=categorized["scale_plot_area_rule"],
            vlm_readability_rules=categorized["vlm_readability_rule"],
            domain_semantics_rules=self._gate_domain_semantics(categorized["domain_semantics_rule"], query_analysis, data_profile),
            applicable_rules=parsed.applicable_rules,
            avoid=parsed.avoid,
            quality_checks=parsed.quality_checks,
            feedback_warnings=parsed.feedback_warnings,
            source_refs=source_refs,
        )
        guidance.prompt_text = self._to_prompt_text(guidance, task_context=task_context)
        return guidance

    @staticmethod
    def _to_prompt_text(guidance: VisRAGGenerationGuidance, task_context: dict[str, Any] | None = None) -> str:
        sections = [
            ("Applicable rules", guidance.applicable_rules),
            ("Avoid", guidance.avoid),
            ("Quality checks", guidance.quality_checks),
            ("Feedback warnings", guidance.feedback_warnings),
        ]
        lines: list[str] = ["VisRAG guidance:"]
        task_block = task_context_prompt_block(task_context)
        if task_block:
            lines.append(task_block)
        for title, items in sections:
            if not items:
                continue
            lines.append(f"{title}:")
            lines.extend(f"- {item}" for item in items[:8])
        if guidance.source_refs:
            lines.append("Source refs:")
            lines.extend(
                f"- {item.get('chunk_id')} ({item.get('source_id')}: {item.get('title')})"
                for item in guidance.source_refs[:8]
            )
        return "\n".join(lines)

    @staticmethod
    def _count_by_kind(chunks: list[VisRAGRetrievedChunk]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.source_kind] = counts.get(chunk.source_kind, 0) + 1
        return counts

    @staticmethod
    def _rule_document_from_chunk(chunk: VisRAGRetrievedChunk | VisRAGGuidanceChunk) -> VisRAGRuleDocument:
        return VisRAGRuleDocument(
            doc_id=chunk.chunk_id,
            record_type=str(chunk.source_kind),
            title=chunk.title,
            retrieval_text=str(chunk.metadata.get("retrieval_text") or chunk.text),
            prompt_text=chunk.text,
            metadata=chunk.metadata,
            score=chunk.score,
        )

    @classmethod
    def _categorized_rule_documents(
            cls,
            chunks: list[VisRAGRetrievedChunk] | list[VisRAGGuidanceChunk],
    ) -> dict[str, list[VisRAGRuleDocument]]:
        result = {
            "chart_pattern": [],
            "readability_rule": [],
            "scale_plot_area_rule": [],
            "vlm_readability_rule": [],
            "domain_semantics_rule": [],
        }
        for chunk in chunks:
            record_type = str(chunk.source_kind)
            if record_type in result:
                result[record_type].append(cls._rule_document_from_chunk(chunk))
        return result

    @staticmethod
    def _domain_semantics_matches(
            chunk: VisRAGGuidanceChunk,
            query_analysis: QueryRequestAnalysisResult,
            data_profile: DataProfile | None,
    ) -> bool:
        query_text = " ".join([
            query_analysis.normalized_query,
            " ".join(query_analysis.selected_fields),
        ]).lower()
        if data_profile is not None:
            query_text += " " + " ".join(column.name for column in data_profile.columns).lower()
        metadata = chunk.metadata or {}
        domain = str(metadata.get("domain") or "").lower()
        if domain in {"medicine", "medical", "biomedical", "biology"}:
            return any(token in query_text for token in ["hba1c", "biomarker", "clinical", "treatment", "cell", "gene", "patient"])
        return True

    @classmethod
    def _gate_domain_semantics(
            cls,
            documents: list[VisRAGRuleDocument],
            query_analysis: QueryRequestAnalysisResult,
            data_profile: DataProfile,
    ) -> list[VisRAGRuleDocument]:
        kept: list[VisRAGRuleDocument] = []
        for document in documents:
            chunk = VisRAGGuidanceChunk(
                chunk_id=document.doc_id,
                source_id=document.doc_id,
                source_kind="domain_semantics_rule",
                title=document.title,
                text=document.prompt_text or document.retrieval_text,
                metadata=document.metadata,
                score=document.score,
            )
            if cls._domain_semantics_matches(chunk, query_analysis, data_profile):
                kept.append(document)
        return kept
