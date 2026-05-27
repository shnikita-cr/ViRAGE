from __future__ import annotations

from pathlib import Path

from src.visrag_core.stores.base import UnsupportedVisRAGStore, VisRAGStore
from src.visrag_core.stores.jsonl_store import JsonlVisRAGStore


def create_visrag_store(*, backend: str, uri: str | Path | None) -> VisRAGStore:
    backend_name = str(backend or "jsonl").strip().lower()
    corpus_uri = Path(uri or "rag_corpus/runtime")
    if backend_name in {"jsonl", "file", "local"}:
        return JsonlVisRAGStore(corpus_uri)
    return UnsupportedVisRAGStore(backend_name=backend_name, corpus_uri=str(corpus_uri))
