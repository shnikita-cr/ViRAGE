from __future__ import annotations

from src.domain.models import VisRAGGuidanceChunk


class VisRAGStore:
    backend_name = "unknown"
    corpus_uri: str | None = None

    def load_chunks(self) -> list[VisRAGGuidanceChunk]:
        raise NotImplementedError

    def load_embeddings(self) -> dict[str, list[float]]:
        raise NotImplementedError

    def corpus_signature(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "uri": self.corpus_uri,
            "exists": False,
            "hash": "missing",
            "cache_key": f"{self.backend_name}:{self.corpus_uri}:missing",
        }


class UnsupportedVisRAGStore(VisRAGStore):
    def __init__(self, *, backend_name: str, corpus_uri: str):
        self.backend_name = backend_name
        self.corpus_uri = corpus_uri

    def load_chunks(self) -> list[VisRAGGuidanceChunk]:
        raise RuntimeError(
            f"Unsupported VisRAG store backend {self.backend_name!r}. "
            "Select an implemented backend or add a VisRAGStore adapter. "
            "Backend choice is intentionally storage-agnostic for future FAISS/Chroma/DB tests."
        )

    def load_embeddings(self) -> dict[str, list[float]]:
        self.load_chunks()
        return {}
