from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.domain.models import VisRAGRuleDocument
from src.visrag_core.stores.base import RuleCorpusRepository


class JsonlRuleCorpusRepository(RuleCorpusRepository):
    """Local JSONL runtime rule store."""

    backend_name = "jsonl"

    def __init__(self, path: Path):
        self.path = path
        self.corpus_uri = str(path)

    def _resolve_file(self) -> Path:
        if self.path.is_file():
            return self.path
        return self.path / "virage_rules.jsonl"

    def corpus_signature(self) -> dict[str, object]:
        corpus_file = self._resolve_file()
        if not corpus_file.exists():
            return {
                "backend": self.backend_name,
                "uri": self.corpus_uri,
                "resolved_path": corpus_file.as_posix(),
                "exists": False,
                "hash": "missing",
                "cache_key": f"jsonl:{corpus_file.as_posix()}:missing",
            }
        stat = corpus_file.stat()
        digest = hashlib.sha256(corpus_file.read_bytes()).hexdigest()
        return {
            "backend": self.backend_name,
            "uri": self.corpus_uri,
            "resolved_path": corpus_file.as_posix(),
            "exists": True,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "hash": digest,
            "cache_key": f"jsonl:{corpus_file.as_posix()}:{stat.st_mtime_ns}:{stat.st_size}:{digest}",
        }

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
