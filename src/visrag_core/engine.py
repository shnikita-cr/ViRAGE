from __future__ import annotations

import json
from dataclasses import dataclass
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
from src.visrag_core.embeddings import build_embedding_model, cosine
from src.visrag_core.query_builder import build_visrag_query
from src.visrag_core.stores import VisRAGStore


@dataclass(frozen=True)
class VisRAGCoreOptions:
    enabled: bool = True
    store_backend: str = "jsonl"
    top_k_chunks: int = 8
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_base_url: str | None = None


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
            embeddings: dict[str, list[float]] | None = None,
            reasoning_llm: Any | None = None,
    ) -> None:
        self.store = store
        self.options = options
        self._signature = corpus_signature or {}
        self._chunks = chunks
        self._embeddings = embeddings
        self.reasoning_llm = reasoning_llm

    def invoke(self, query_analysis: QueryRequestAnalysisResult, data_profile: DataProfile) -> VisRAGResult:
        if not self.options.enabled:
            raise RuntimeError("VisRAG is disabled. Enable visrag_enabled=true or remove the visrag pipeline node.")
        chunks = self._chunks if self._chunks is not None else self.store.load_chunks()
        embeddings = self._embeddings if self._embeddings is not None else self.store.load_embeddings()
        query = build_visrag_query(query_analysis, data_profile)
        if not embeddings:
            raise RuntimeError(
                "VisRAG embeddings are missing. Runtime lexical fallback is disabled by design. "
                "Rebuild runtime embeddings before running the pipeline:\n"
                "python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model bge-m3:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600"
            )
        missing = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in embeddings]
        if missing:
            raise RuntimeError(
                "VisRAG embeddings are incomplete. Runtime lexical fallback is disabled by design. "
                "Rebuild runtime embeddings before running the pipeline:\n"
                "python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model bge-m3:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600\n"
                f"Missing embeddings for {len(missing)} chunks; first missing chunk_id={missing[0]!r}."
            )
        retrieved = self._retrieve(query, query_analysis, chunks, embeddings)
        guidance = self._generate_response(query_analysis, data_profile, retrieved)
        diagnostics = VisRAGDiagnostics(
            retrieved_count=len(retrieved),
            retrieved_count_by_type=self._count_by_kind(retrieved),
            corpus_backend=self.store.backend_name,
            corpus_uri=self.store.corpus_uri,
            corpus_hash=str(self._signature.get("hash") or ""),
        )
        debug = VisRAGDebugRetrieval(
            retrieval_query=query,
            retrieved_chunks=retrieved,
            retrieved_documents=retrieved,
            scores=[{"chunk_id": chunk.chunk_id, "score": chunk.score, "source_id": chunk.source_id} for chunk in retrieved],
        )
        return VisRAGResult(
            corpus_status={
                "enabled": True,
                "chunks": len(chunks),
                "backend": self.store.backend_name,
                "signature": self._signature,
            },
            retrieval_strategy=f"chunk_guidance:{self.store.backend_name}:precomputed_embeddings",
            generation_guidance=guidance,
            debug_retrieval=debug,
            diagnostics=diagnostics,
        )

    def _retrieve(
            self,
            query: str,
            query_analysis: QueryRequestAnalysisResult,
            chunks: list[VisRAGGuidanceChunk],
            embeddings: dict[str, list[float]],
    ) -> list[VisRAGRetrievedChunk]:
        if not query.strip():
            return []
        embedder = build_embedding_model(
            provider=self.options.embedding_provider,
            model=self.options.embedding_model,
            base_url=self.options.embedding_base_url,
        )
        query_vector = list(embedder.embed_query(query))
        ranked: list[VisRAGRetrievedChunk] = []
        for chunk in chunks:
            score = cosine(query_vector, embeddings[chunk.chunk_id])
            score *= self._metadata_weight(chunk)
            if score > 0:
                ranked.append(VisRAGRetrievedChunk(**chunk.model_dump(exclude={"score"}), score=round(score, 6)))
        return sorted(ranked, key=lambda item: (-item.score, item.source_id, item.chunk_id))[:max(1, self.options.top_k_chunks)]


    @staticmethod
    def _metadata_weight(chunk: VisRAGGuidanceChunk) -> float:
        metadata = chunk.metadata or {}
        weight = float(metadata.get("priority", 1.0) or 1.0)
        if chunk.source_kind in {"manual_feedback", "vlm_feedback"}:
            weight *= 1.5
        return max(0.1, min(weight, 4.0))

    def _generate_response(
            self,
            query_analysis: QueryRequestAnalysisResult,
            data_profile: DataProfile,
            chunks: list[VisRAGRetrievedChunk],
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
            guidance.prompt_text = self._to_prompt_text(guidance)
            return guidance

        payload = {
            "query_analysis": {
                "normalized_query": query_analysis.normalized_query,
                "analysis_task": query_analysis.analysis_task,
                "selected_fields": query_analysis.selected_fields,
            },
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
        prompt = (
            "Generate final VisRAG guidance for Vega-Lite spec generation. "
            "Use only the retrieved chunks and the given data/query context. "
            "Do not output Vega-Lite code or examples. Return concrete practical guidance.\n\n"
            f"Context JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
        )
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
        guidance.prompt_text = self._to_prompt_text(guidance)
        return guidance

    @staticmethod
    def _to_prompt_text(guidance: VisRAGGenerationGuidance) -> str:
        sections = [
            ("Applicable rules", guidance.applicable_rules),
            ("Avoid", guidance.avoid),
            ("Quality checks", guidance.quality_checks),
            ("Feedback warnings", guidance.feedback_warnings),
        ]
        lines: list[str] = ["VisRAG guidance:"]
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

