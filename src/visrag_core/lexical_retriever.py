from __future__ import annotations

from src.domain.models import VisRAGGuidanceChunk
from src.visrag_core.text import tokens


class RankBM25ChunkRetriever:
    backend_name = "lexical"

    def score(self, *, query: str, chunks: list[VisRAGGuidanceChunk]) -> dict[str, float]:
        query_tokens = tokens(query)
        if not query_tokens:
            return {chunk.chunk_id: 0.0 for chunk in chunks}
        tokenized_docs = [tokens(self._chunk_retrieval_text(chunk)) for chunk in chunks]
        if not tokenized_docs:
            return {}
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("VisRAG lexical backend requires the rank-bm25 package.") from exc
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(query_tokens)
        return {chunk.chunk_id: float(score) for chunk, score in zip(chunks, scores, strict=False)}

    @staticmethod
    def _chunk_retrieval_text(chunk: VisRAGGuidanceChunk) -> str:
        metadata = chunk.metadata or {}
        metadata_text = " ".join(str(value) for value in metadata.values() if isinstance(value, (str, int, float)))
        return " ".join([chunk.title, chunk.source_name, chunk.source_kind, metadata_text, chunk.text])
