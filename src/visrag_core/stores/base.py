from __future__ import annotations

from src.domain.models import VisRAGRuleDocument


class RuleCorpusRepository:
    """Storage boundary for runtime VisRAG rule/guidance documents.

    Implementations may read JSONL, a vector database, a graph store, or any
    other backend. Downstream VisRAG logic depends only on this interface.
    """

    backend_name = "unknown"
    corpus_uri: str | None = None

    def load_documents(self) -> list[VisRAGRuleDocument]:
        raise NotImplementedError


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
