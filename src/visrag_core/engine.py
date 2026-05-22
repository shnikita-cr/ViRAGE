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
from src.visrag_core.filters import domain_semantics_gate
from src.visrag_core.query_builder import build_typed_queries
from src.visrag_core.retrievers import BM25LikeRuleRetriever
from src.visrag_core.stores import RuleCorpusRepository


@dataclass(frozen=True)
class VisRAGCoreOptions:
    enabled: bool = True
    retriever_name: str = "bm25"
    top_k_by_type: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TOP_K))
    disabled_message: str = "VisRAG guidance is disabled."

    def top_k(self, record_type: str) -> int:
        return max(0, int(self.top_k_by_type.get(record_type, DEFAULT_TOP_K.get(record_type, 1))))


class VisRAGEngine:
    """Core runtime VisRAG retrieval and guidance composition.

    The engine owns typed retrieval, domain gating, diagnostics, and guidance
    composition. It does not know about graph nodes, application services, or
    concrete LLM calls. Storage is injected through RuleCorpusRepository.
    """

    def __init__(self, repository: RuleCorpusRepository, options: VisRAGCoreOptions | None = None) -> None:
        self.repository = repository
        self.options = options or VisRAGCoreOptions()
        self._retriever = BM25LikeRuleRetriever()

    def invoke(
            self,
            query_analysis: QueryRequestAnalysisResult,
            data_profile: DataProfile,
    ) -> VisRAGResult:
        if not self.options.enabled:
            guidance = VisRAGGenerationGuidance(prompt_text=self.options.disabled_message)
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

        raw_documents = self.repository.load_documents()
        if not raw_documents:
            diagnostics = VisRAGDiagnostics(
                warnings=["No runtime rule documents were found."],
                corpus_backend=self.repository.backend_name,
                corpus_uri=self.repository.corpus_uri,
            )
            return VisRAGResult(
                caveats=["No runtime rule documents were found."],
                corpus_status={"enabled": True, "documents": 0, "backend": self.repository.backend_name},
                retrieval_strategy=f"rule_guidance:{self.repository.backend_name}",
                generation_guidance=VisRAGGenerationGuidance(prompt_text="No VisRAG guidance was retrieved."),
                diagnostics=diagnostics,
            )

        queries = build_typed_queries(query_analysis, data_profile)
        selected_by_type: dict[str, list[VisRAGRuleDocument]] = {}
        all_selected: list[VisRAGRuleDocument] = []
        filtered: list[dict[str, Any]] = []

        for record_type in RULE_TYPES:
            top_k = self.options.top_k(record_type)
            if top_k <= 0:
                selected_by_type[record_type] = []
                continue
            if record_type == "domain_semantics_rule" and not domain_semantics_gate(query_analysis, data_profile):
                selected_by_type[record_type] = []
                continue
            docs = [doc for doc in raw_documents if doc.record_type == record_type]
            ranked = self._retriever.rank(
                docs,
                query=queries.get(record_type) or query_analysis.normalized_query,
                query_analysis=query_analysis,
            )
            picked = ranked[:top_k]
            selected_by_type[record_type] = picked
            all_selected.extend(picked)
            if len(ranked) > top_k:
                filtered.extend([
                    {"doc_id": doc.doc_id, "record_type": doc.record_type, "reason": "below_top_k", "score": doc.score}
                    for doc in ranked[top_k:top_k + 5]
                ])

        guidance = compose_generation_guidance(selected_by_type, query_analysis)
        diagnostics = VisRAGDiagnostics(
            retrieved_count_by_type={key: len(value) for key, value in selected_by_type.items()},
            corpus_backend=self.repository.backend_name,
            corpus_uri=self.repository.corpus_uri,
            warnings=[],
        )
        debug = VisRAGDebugRetrieval(
            retrieval_queries=queries,
            retrieved_documents=all_selected,
            filtered_documents=filtered,
        )
        return VisRAGResult(
            caveats=[],
            corpus_status={"enabled": True, "documents": len(raw_documents), "backend": self.repository.backend_name},
            retrieval_strategy=f"rule_guidance:{self.repository.backend_name}:{self.options.retriever_name}",
            generation_guidance=guidance,
            debug_retrieval=debug,
            diagnostics=diagnostics,
        )
