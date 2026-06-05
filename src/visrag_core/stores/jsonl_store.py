from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.domain.models import VisRAGGuidanceChunk
from src.visrag_core.stores.base import VisRAGStore


class JsonlVisRAGStore(VisRAGStore):
    backend_name = "jsonl"

    def __init__(self, uri: Path):
        self.root = uri
        self.corpus_uri = str(uri)

    @property
    def chunks_path(self) -> Path:
        if self.root.is_file():
            return self.root
        return self.root / "guidance_chunks.jsonl"

    def corpus_signature(self) -> dict[str, object]:
        digest = hashlib.sha256()
        if self.chunks_path.exists():
            digest.update(self.chunks_path.name.encode("utf-8"))
            digest.update(self.chunks_path.read_bytes())
            content_hash = digest.hexdigest()
        else:
            content_hash = "missing"
        return {
            "backend": self.backend_name,
            "uri": self.corpus_uri,
            "chunks_path": self.chunks_path.as_posix(),
            "chunks_exists": self.chunks_path.exists(),
            "hash": content_hash,
            "cache_key": f"jsonl:{self.chunks_path}:{content_hash}",
        }

    def load_chunks(self) -> list[VisRAGGuidanceChunk]:
        if not self.chunks_path.exists():
            raise RuntimeError(
                "VisRAG guidance chunks are missing. Run:\n"
                "python scripts\\rag_corpus\\export_guidance_chunks.py"
            )
        chunks: list[VisRAGGuidanceChunk] = []
        with self.chunks_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                raw.setdefault("metadata", {})["line_number"] = line_number
                chunks.append(VisRAGGuidanceChunk.model_validate(raw))
        return chunks
