from __future__ import annotations

from pathlib import Path

from src.visrag_core.stores.base import RuleCorpusRepository, UnsupportedRuleCorpusRepository
from src.visrag_core.stores.jsonl_store import JsonlRuleCorpusRepository


def create_rule_corpus_repository(*, backend: str, uri: str | Path | None) -> RuleCorpusRepository:
    backend_name = str(backend or "jsonl").strip().lower()
    corpus_uri = Path(uri or "rag_corpus/runtime")
    if backend_name in {"jsonl", "file", "local"}:
        return JsonlRuleCorpusRepository(corpus_uri)
    return UnsupportedRuleCorpusRepository(backend_name=backend_name, corpus_uri=str(corpus_uri))
