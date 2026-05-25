from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.models import (
    DataProfile,
    QueryRequestAnalysisResult,
    VisRAGDebugRetrieval,
    VisRAGDiagnostics,
    VisRAGGenerationGuidance,
    VisRAGResult,
    VisRAGRuleDocument,
)
from src.visrag_core.composer import compose_generation_guidance
from src.visrag_core.constants import DEFAULT_TOP_K, RULE_TYPES
from src.visrag_core.filters import domain_semantics_gate, rerank_by_compatibility
from src.visrag_core.query_builder import build_typed_queries
from src.visrag_core.rule_retrieval import RuleRetriever, RuleRetrieverOptions, build_rule_retriever
from src.visrag_core.stores import RuleCorpusRepository


@dataclass(frozen=True)
class VisRAGCoreOptions:
    enabled: bool = True
    retriever_name: str = "bm25"
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    top_k_by_type: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TOP_K))
    disabled_message: str = ""

    def top_k(self, record_type: str) -> int:
        return max(0, int(self.top_k_by_type.get(record_type, DEFAULT_TOP_K.get(record_type, 1))))

    def retriever_options(self) -> RuleRetrieverOptions:
        return RuleRetrieverOptions(
            backend=self.retriever_name,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_base_url=self.embedding_base_url,
        )


class VisRAGEngine:
    """Core runtime VisRAG retrieval and guidance composition.

    The engine retrieves rule/guidance documents only. It does not materialize or
    rank Vega-Lite specification templates. Concrete storage, corpus caching and
    retriever instance reuse are injected by the application service.
    """

    def __init__(
            self,
            repository: RuleCorpusRepository,
            options: VisRAGCoreOptions | None = None,
            *,
            documents: list[VisRAGRuleDocument] | None = None,
            retriever: RuleRetriever | None = None,
            corpus_signature: dict[str, Any] | None = None,
    ) -> None:
        self.repository = repository
        self.options = options or VisRAGCoreOptions()
        self._documents = documents
        self._retriever = retriever or build_rule_retriever(self.options.retriever_options())
        self._corpus_signature = corpus_signature or {}

    def invoke(
            self,
            query_analysis: QueryRequestAnalysisResult,
            data_profile: DataProfile,
    ) -> VisRAGResult:
        if not self.options.enabled:
            guidance = VisRAGGenerationGuidance(prompt_text="")
            diagnostics = VisRAGDiagnostics(
                warnings=["VisRAG disabled by project settings."],
                corpus_backend="disabled",
            )
            return VisRAGResult(
                caveats=["VisRAG disabled by project settings."],
                corpus_status={"enabled": False, "documents": 0},
                retrieval_strategy="disabled",
                generation_guidance=guidance,
                diagnostics=diagnostics,
            )

        raw_documents = self._documents if self._documents is not None else self.repository.load_documents()
        if not raw_documents:
            diagnostics = VisRAGDiagnostics(
                warnings=["No runtime rule documents were found."],
                corpus_backend=self.repository.backend_name,
                corpus_uri=self.repository.corpus_uri,
                corpus_hash=str(self._corpus_signature.get("hash") or ""),
            )
            return VisRAGResult(
                caveats=["No runtime rule documents were found."],
                corpus_status={
                    "enabled": True,
                    "documents": 0,
                    "backend": self.repository.backend_name,
                    "signature": self._corpus_signature,
                },
                retrieval_strategy=f"rule_guidance:{self.repository.backend_name}",
                generation_guidance=VisRAGGenerationGuidance(prompt_text=""),
                diagnostics=diagnostics,
            )

        queries = build_typed_queries(query_analysis, data_profile)
        selected_by_type: dict[str, list[VisRAGRuleDocument]] = {}
        all_selected: list[VisRAGRuleDocument] = []
        filtered: list[dict[str, Any]] = []
        scores_by_type: dict[str, list[dict[str, Any]]] = {}
        corpus_key = str(self._corpus_signature.get("hash") or self._corpus_signature.get("cache_key") or "")

        for record_type in RULE_TYPES:
            top_k = self.options.top_k(record_type)
            if top_k <= 0:
                selected_by_type[record_type] = []
                scores_by_type[record_type] = []
                continue
            if record_type == "domain_semantics_rule" and not domain_semantics_gate(query_analysis, data_profile):
                selected_by_type[record_type] = []
                scores_by_type[record_type] = []
                continue
            docs = [doc for doc in raw_documents if doc.record_type == record_type]
            ranked = self._retriever.rank(
                docs,
                query=queries.get(record_type) or query_analysis.normalized_query,
                query_analysis=query_analysis,
                corpus_key=f"{corpus_key}:{record_type}",
            )
            reranked, compatibility_filtered = rerank_by_compatibility(ranked, query_analysis, data_profile)
            filtered.extend(compatibility_filtered)
            picked = reranked[:top_k]
            selected_by_type[record_type] = picked
            all_selected.extend(picked)
            scores_by_type[record_type] = [
                {
                    "doc_id": doc.doc_id,
                    "score": doc.score,
                    "source": (doc.metadata or {}).get("source_dataset") or (doc.metadata or {}).get("source"),
                }
                for doc in reranked[:max(top_k, 5)]
            ]
            if len(reranked) > top_k:
                filtered.extend([
                    {"doc_id": doc.doc_id, "record_type": doc.record_type, "reason": "below_top_k", "score": doc.score}
                    for doc in reranked[top_k:top_k + 5]
                ])

        guidance = compose_generation_guidance(selected_by_type, query_analysis)
        compatibility_filter_count = sum(
            1 for item in filtered if str(item.get("reason", "")).startswith(("incompatible_chart_family", "missing_required_data"))
        )
        diagnostics = VisRAGDiagnostics(
            retrieved_count_by_type={key: len(value) for key, value in selected_by_type.items()},
            corpus_backend=self.repository.backend_name,
            corpus_uri=self.repository.corpus_uri,
            corpus_hash=str(self._corpus_signature.get("hash") or ""),
            warnings=[
                f"Compatibility reranker removed {compatibility_filter_count} incompatible retrieved rules."
            ] if compatibility_filter_count else [],
        )
        debug = VisRAGDebugRetrieval(
            retrieval_queries=queries,
            retrieved_documents=all_selected,
            filtered_documents=filtered,
            scores_by_type=scores_by_type,
        )
        return VisRAGResult(
            caveats=[],
            corpus_status={
                "enabled": True,
                "documents": len(raw_documents),
                "backend": self.repository.backend_name,
                "signature": self._corpus_signature,
            },
            retrieval_strategy=f"rule_guidance:{self.repository.backend_name}:{self.options.retriever_name}",
            generation_guidance=guidance,
            debug_retrieval=debug,
            diagnostics=diagnostics,
        )
