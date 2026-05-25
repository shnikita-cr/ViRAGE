from __future__ import annotations

from src.domain.models import VisRAGRuleDocument


class RuleCorpusRepository:
    """Storage boundary for runtime VisRAG rule/guidance documents."""

    backend_name = "unknown"
    corpus_uri: str | None = None

    def load_documents(self) -> list[VisRAGRuleDocument]:
        raise NotImplementedError

    def corpus_signature(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "uri": self.corpus_uri,
            "exists": False,
            "hash": "missing",
            "cache_key": f"{self.backend_name}:{self.corpus_uri}:missing",
        }


class UnsupportedRuleCorpusRepository(RuleCorpusRepository):
    def __init__(self, *, backend_name: str, corpus_uri: str):
        self.backend_name = backend_name
        self.corpus_uri = corpus_uri

    def load_documents(self) -> list[VisRAGRuleDocument]:
        raise RuntimeError(
            f"Unsupported VisRAG runtime store backend {self.backend_name!r}. "
            "Current runtime implementation supports local JSONL exports. "
            "Add a new RuleCorpusRepository implementation for vector or graph storage."
        )
