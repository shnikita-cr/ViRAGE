from __future__ import annotations

from src.domain.models import VisRAGGuidanceChunk, VisRAGRetrievedChunk


def test_retrieved_chunk_is_not_plain_guidance_chunk() -> None:
    chunk = VisRAGGuidanceChunk(
        chunk_id="c1",
        source_id="source",
        source_kind="web_guidance",
        title="Test",
        text="Useful visualization guidance.",
    )
    retrieved = VisRAGRetrievedChunk(**chunk.model_dump(exclude={"score"}), score=0.9)
    assert isinstance(retrieved, VisRAGRetrievedChunk)
    assert retrieved.score == 0.9
