from __future__ import annotations

from src.domain.models import DataColumnProfile, DataProfile, QueryRequestAnalysisResult, VisRAGGuidanceChunk
from src.visrag_core.engine import VisRAGCoreOptions, VisRAGEngine
from src.visrag_core.query_builder import build_visrag_query
from src.visrag_core.task_context import task_context_from_user_context
from src.visrag_core.stores.base import VisRAGStore


class _Store(VisRAGStore):
    backend_name = "test"
    corpus_uri = "memory"

    def load_chunks(self):
        return []

    def load_embeddings(self):
        return {}

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


def test_runtime_visrag_requires_embeddings_without_lexical_fallback() -> None:
    engine = VisRAGEngine(
        store=_Store(),
        options=VisRAGCoreOptions(enabled=True),
        chunks=[VisRAGGuidanceChunk(chunk_id="c1", source_id="s", text="Use box plots for group comparison.")],
        embeddings={},
    )

    try:
        engine.invoke(_analysis(), _profile())
    except RuntimeError as exc:
        assert "Runtime lexical fallback is disabled" in str(exc)
    else:
        raise AssertionError("Expected runtime RAG to fail without embeddings.")


def test_task_contract_is_written_to_guidance_prompt(monkeypatch) -> None:
    monkeypatch.setattr("src.visrag_core.engine.build_embedding_model", lambda **kwargs: _FakeEmbedder())
    engine = VisRAGEngine(
        store=_Store(),
        options=VisRAGCoreOptions(enabled=True, top_k_chunks=1),
        chunks=[
            VisRAGGuidanceChunk(
                chunk_id="c1",
                source_id="scientific",
                source_kind="scientific_figure_guidance",
                title="Group comparison",
                text="For group comparisons, show uncertainty and keep axis units visible.",
            )
        ],
        embeddings={"c1": [1.0, 0.0]},
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
