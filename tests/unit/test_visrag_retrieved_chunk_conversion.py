from __future__ import annotations

from src.domain.models import DataProfile, QueryRequestAnalysisResult, VisRAGGuidanceChunk, VisRAGRetrievedChunk
from src.visrag_core.engine import VisRAGCoreOptions, VisRAGEngine


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_retrieve_converts_guidance_chunks_to_retrieved_chunks(monkeypatch) -> None:
    import src.visrag_core.engine as engine_module

    monkeypatch.setattr(engine_module, "build_embedding_model", lambda **_: _FakeEmbedder())
    engine = VisRAGEngine(store=object(), options=VisRAGCoreOptions(top_k_chunks=2))
    query_analysis = QueryRequestAnalysisResult(normalized_query="compare categories", recommended_chart_family="bar_chart")
    chunks = [
        VisRAGGuidanceChunk(
            chunk_id="chunk-1",
            source_id="wilke_fundamentals",
            source_name="Wilke",
            title="Compare amounts",
            text="Use readable category comparison.",
            metadata={"chart_family": "bar_chart"},
        )
    ]
    retrieved = engine._retrieve("compare categories", query_analysis, chunks, {"chunk-1": [1.0, 0.0]})

    assert len(retrieved) == 1
    assert isinstance(retrieved[0], VisRAGRetrievedChunk)
    assert retrieved[0].chunk_id == "chunk-1"
    assert retrieved[0].score > 0
