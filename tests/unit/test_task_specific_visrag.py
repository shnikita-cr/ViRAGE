from __future__ import annotations

from src.domain.models import DataColumnProfile, DataProfile, QueryRequestAnalysisResult, VisRAGGuidanceChunk
from src.visrag_core.engine import VisRAGCoreOptions, VisRAGEngine
from src.visrag_core.retrieval.rules.query_builder import build_visrag_query
from src.visrag_core.models.task_context import task_context_from_user_context
from src.visrag_core.stores.base import VisRAGStore


class _Store(VisRAGStore):
    backend_name = "test"
    corpus_uri = "memory"

    def load_chunks(self):
        return []

    def corpus_signature(self):
        return {"hash": "test"}


class _FakeEmbedder:
    def embed_query(self, query: str):
        return [1.0, 0.0]


def _profile() -> DataProfile:
    return DataProfile(
        row_count=20,
        col_count=2,
        columns=[
            DataColumnProfile(name="condition", dtype="object", role="dimension"),
            DataColumnProfile(name="score", dtype="float", role="measure"),
        ],
    )


def _analysis() -> QueryRequestAnalysisResult:
    return QueryRequestAnalysisResult(
        normalized_query="Compare score across condition.",
        analysis_task="comparison",
        selected_fields=["condition", "score"],
    )


def test_task_context_is_added_to_visrag_query() -> None:
    task_context = {
        "task_type": "group_comparison",
        "purpose": "Compare a numeric measurement between experimental conditions.",
        "required_fields": ["condition", "score"],
        "constraints": {"output_target": "scientific_figure", "task_is_fixed": True},
    }

    query = build_visrag_query(_analysis(), _profile(), task_context=task_context)

    assert "group_comparison" in query
    assert "condition score" in query
    assert "scientific figure" in query
    assert "auto" not in query


def test_user_context_extraction_keeps_orchestrator_subtask_contract() -> None:
    context = task_context_from_user_context(
        {
            "analysis_subtask": {
                "id": "group_comparison_001",
                "task_type": "group_comparison",
                "query": "Compare score across condition.",
                "purpose": "Compare groups.",
                "required_fields": ["condition", "score"],
                "constraints": {},
            }
        }
    )

    assert context["task_type"] == "group_comparison"
    assert context["constraints"]["output_target"] == "scientific_figure"
    assert context["constraints"]["task_is_fixed"] is True


def test_task_contract_is_written_to_guidance_prompt(monkeypatch) -> None:
    monkeypatch.setattr("src.visrag_core.engine.VisRAGEngine._validate_semantic_index", lambda self, chunks: None)
    monkeypatch.setattr("src.visrag_core.engine.ChromaChunkRetriever.score", lambda self, query, chunks, limit: {"c1": 1.0})
    engine = VisRAGEngine(
        store=_Store(),
        options=VisRAGCoreOptions(enabled=True, retrieval_backend="semantic", top_k_chunks=1),
        chunks=[
            VisRAGGuidanceChunk(
                chunk_id="c1",
                source_id="scientific",
                source_kind="scientific_figure_guidance",
                title="Group comparison",
                text="For group comparisons, show uncertainty and keep axis units visible.",
            )
        ],
    )

    result = engine.invoke(
        _analysis(),
        _profile(),
        task_context={
            "task_type": "group_comparison",
            "purpose": "Compare score between conditions.",
            "required_fields": ["condition", "score"],
            "constraints": {"output_target": "scientific_figure", "task_is_fixed": True},
        },
    )

    prompt = result.generation_guidance.prompt_text
    assert "Selected analytical task contract" in prompt
    assert "group_comparison" in prompt
    assert "Do not replace this task" in prompt
    assert result.debug_retrieval.task_context["task_type"] == "group_comparison"


def test_runtime_visrag_supports_explicit_semantic_backend(monkeypatch) -> None:
    monkeypatch.setattr("src.visrag_core.engine.VisRAGEngine._validate_semantic_index", lambda self, chunks: None)
    monkeypatch.setattr("src.visrag_core.engine.ChromaChunkRetriever.score", lambda self, query, chunks, limit: {"c1": 1.0, "c2": 0.0})
    engine = VisRAGEngine(
        store=_Store(),
        options=VisRAGCoreOptions(enabled=True, retrieval_backend="semantic", top_k_chunks=1),
        chunks=[
            VisRAGGuidanceChunk(chunk_id="c1", source_id="s", text="Use compact boxplots for group comparison."),
            VisRAGGuidanceChunk(chunk_id="c2", source_id="s", text="Use maps for geographic coordinates."),
        ],
    )

    result = engine.invoke(_analysis(), _profile())

    assert result.retrieval_strategy.startswith("chunk_guidance:semantic:chroma")
    assert [chunk.chunk_id for chunk in result.debug_retrieval.retrieved_chunks] == ["c1"]


def test_runtime_visrag_supports_explicit_hybrid_backend(monkeypatch) -> None:
    monkeypatch.setattr("src.visrag_core.engine.VisRAGEngine._validate_semantic_index", lambda self, chunks: None)
    monkeypatch.setattr("src.visrag_core.engine.ChromaChunkRetriever.score", lambda self, query, chunks, limit: {"c1": 0.1, "c2": 1.0})
    engine = VisRAGEngine(
        store=_Store(),
        options=VisRAGCoreOptions(
            enabled=True,
            retrieval_backend="hybrid",
            top_k_chunks=1,
            hybrid_method="cc",
            hybrid_weight=0.1,
        ),
        chunks=[
            VisRAGGuidanceChunk(chunk_id="c1", source_id="s", text="Use compact boxplots for group comparison."),
            VisRAGGuidanceChunk(chunk_id="c2", source_id="s", text="Unrelated geographic map guidance."),
        ],
    )

    monkeypatch.setattr(
        "src.visrag_core.engine.RankBM25ChunkRetriever.score",
        lambda self, query, chunks: {"c1": 2.0, "c2": 0.0},
    )
    result = engine.invoke(_analysis(), _profile())

    assert result.retrieval_strategy.startswith("chunk_guidance:hybrid:chroma+rank_bm25:cc")
    assert result.debug_retrieval.retrieved_chunks[0].chunk_id == "c1"
    assert result.debug_retrieval.retrieved_chunks[0].metadata["retrieval_score_kind"] == "hybrid"


def test_runtime_visrag_supports_explicit_lexical_backend_without_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.visrag_core.engine.RankBM25ChunkRetriever.score",
        lambda self, query, chunks: {"c1": 2.0, "c2": 0.0},
    )
    engine = VisRAGEngine(
        store=_Store(),
        options=VisRAGCoreOptions(enabled=True, retrieval_backend="lexical", top_k_chunks=1),
        chunks=[
            VisRAGGuidanceChunk(chunk_id="c1", source_id="s", text="Use compact boxplots for group comparison."),
            VisRAGGuidanceChunk(chunk_id="c2", source_id="s", text="Use maps for geographic coordinates."),
        ],
    )

    result = engine.invoke(_analysis(), _profile())

    assert result.retrieval_strategy.startswith("chunk_guidance:lexical:rank_bm25")
    assert [chunk.chunk_id for chunk in result.debug_retrieval.retrieved_chunks] == ["c1"]
    assert result.debug_retrieval.retrieved_chunks[0].metadata["retrieval_score_kind"] == "lexical"
