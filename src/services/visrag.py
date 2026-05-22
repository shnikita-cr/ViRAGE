from __future__ import annotations

from src.domain.models import DataProfile, QueryRequestAnalysisResult, VisRAGResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.visrag_core import VisRAGCoreOptions, VisRAGEngine, create_rule_corpus_repository
from src.visrag_core.constants import DEFAULT_TOP_K


class VisRAGService(BaseService):
    """Application service wrapper around the standalone VisRAG core.

    Runtime VisRAG retrieves rule/guidance documents only. It does not return
    Vega-Lite specification candidates or templates. Storage and retrieval logic
    live in src/visrag_core so JSONL can later be replaced with a vector DB or a
    graph store without changing graph nodes or ChartGeneratorService.
    """

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
            top_k_by_type=self._top_k_by_type(runtime),
        )
        return VisRAGEngine(repository=repository, options=options).invoke(query_analysis, data_profile)

    @staticmethod
    def _top_k_by_type(runtime: RuntimeContext) -> dict[str, int]:
        values: dict[str, int] = {}
        for record_type, default_value in DEFAULT_TOP_K.items():
            setting_name = f"visrag_top_k_{record_type}s"
            value = getattr(runtime.settings, setting_name, None)
            values[record_type] = max(0, int(value if value is not None else default_value))
        return values
