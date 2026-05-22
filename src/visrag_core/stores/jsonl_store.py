from __future__ import annotations

import json
from pathlib import Path

from src.domain.models import VisRAGRuleDocument
from src.visrag_core.stores.base import RuleCorpusRepository


class JsonlRuleCorpusRepository(RuleCorpusRepository):
    """Local JSONL runtime rule store.

    This is the simplest current backend. It intentionally implements the same
    repository interface that a future vector database or graph store should
    implement, so service and generator code do not depend on the file format.
    """

    backend_name = "jsonl"

    def __init__(self, path: Path):
        self.path = path
        self.corpus_uri = str(path)

    def _resolve_file(self) -> Path:
        if self.path.is_file():
            return self.path
        return self.path / "virage_rules.jsonl"

    def load_documents(self) -> list[VisRAGRuleDocument]:
        corpus_file = self._resolve_file()
        if not corpus_file.exists():
            return []
        documents: list[VisRAGRuleDocument] = []
        with corpus_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                raw = json.loads(stripped)
                metadata = dict(raw.get("metadata") or {})
                title = metadata.get("title") or raw.get("title") or raw.get("doc_id") or ""
                documents.append(VisRAGRuleDocument(
                    doc_id=str(raw["doc_id"]),
                    record_type=raw["record_type"],
                    title=str(title),
                    retrieval_text=str(raw.get("retrieval_text") or ""),
                    prompt_text=str(raw.get("prompt_text") or ""),
                    metadata=metadata | {"line_number": line_number},
                ))
        return documents
