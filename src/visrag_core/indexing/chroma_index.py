from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.domain.models import VisRAGGuidanceChunk
from src.visrag_core.indexing.embeddings import build_embedding_model

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def chunk_embedding_text(chunk: VisRAGGuidanceChunk, *, max_chars: int = 1600) -> str:
    metadata = chunk.metadata or {}
    metadata_terms = " ".join(str(value) for value in metadata.values() if isinstance(value, (str, int, float, bool)))
    text = "\n".join(
        item.strip()
        for item in [chunk.title, chunk.source_name, chunk.source_kind, metadata_terms, chunk.text]
        if str(item or "").strip()
    )
    return text[:max_chars]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chroma_manifest_path(persist_dir: Path) -> Path:
    return persist_dir / MANIFEST_FILENAME


def normalize_chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[str(key)] = value
            continue
        normalized[str(key)] = json.dumps(value, ensure_ascii=False, default=str)
    return normalized


def validate_chunks_for_index(chunks: list[VisRAGGuidanceChunk]) -> None:
    if not chunks:
        raise RuntimeError("VisRAG corpus has no guidance chunks.")
    seen: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = str(chunk.chunk_id or "").strip()
        if not chunk_id:
            raise RuntimeError(f"Guidance chunk #{index} has empty chunk_id.")
        if chunk_id in seen:
            raise RuntimeError(f"Duplicate guidance chunk_id: {chunk_id}")
        seen.add(chunk_id)
        if not str(chunk.text or "").strip():
            raise RuntimeError(f"Guidance chunk {chunk_id!r} has empty text.")


def write_chroma_manifest(
        *,
        persist_dir: Path,
        chunks_path: Path,
        collection_name: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        chunk_count: int,
) -> dict[str, Any]:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "corpus_path": chunks_path.as_posix(),
        "corpus_sha256": file_sha256(chunks_path),
        "chunk_count": int(chunk_count),
        "collection_name": collection_name,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_dimension": int(embedding_dimension),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    persist_dir.mkdir(parents=True, exist_ok=True)
    chroma_manifest_path(persist_dir).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def read_chroma_manifest(persist_dir: Path) -> dict[str, Any]:
    path = chroma_manifest_path(persist_dir)
    if not path.is_file():
        raise RuntimeError(f"VisRAG Chroma manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid VisRAG Chroma manifest: {path}")
    return payload


def validate_chroma_manifest(
        *,
        persist_dir: Path,
        chunks_path: Path,
        collection_name: str,
        embedding_provider: str | None,
        embedding_model: str | None,
        expected_chunk_count: int,
) -> dict[str, Any]:
    manifest = read_chroma_manifest(persist_dir)
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "collection_name": collection_name,
        "embedding_provider": embedding_provider or "ollama",
        "embedding_model": embedding_model or "nomic-embed-text:latest",
        "corpus_sha256": file_sha256(chunks_path),
        "chunk_count": int(expected_chunk_count),
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "VisRAG Chroma index does not match the current runtime configuration: "
            + json.dumps(mismatches, ensure_ascii=False, default=str)
        )
    return manifest


class ChromaChunkRetriever:
    def __init__(
            self,
            *,
            persist_dir: Path,
            collection_name: str,
            embedding_provider: str | None,
            embedding_model: str | None,
            embedding_base_url: str | None,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider or "ollama"
        self.embedding_model = embedding_model or "nomic-embed-text:latest"
        self.embedding_base_url = embedding_base_url

    def score(self, *, query: str, chunks: list[VisRAGGuidanceChunk], limit: int) -> dict[str, float]:
        if not query.strip():
            return {}
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("VisRAG semantic retrieval requires the chromadb package.") from exc
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if not self.persist_dir.exists():
            raise RuntimeError(f"VisRAG Chroma index directory is missing: {self.persist_dir}")
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection = client.get_collection(name=self.collection_name)
        embedder = build_embedding_model(
            provider=self.embedding_provider,
            model=self.embedding_model,
            base_url=self.embedding_base_url,
        )
        query_vector = list(embedder.embed_query(query))
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=max(1, min(int(limit), len(chunks))),
            include=["distances"],
        )
        ids = result.get("ids") or [[]]
        distances = result.get("distances") or [[]]
        scores: dict[str, float] = {}
        for chunk_id, distance in zip(ids[0], distances[0], strict=False):
            key = str(chunk_id)
            if key not in chunk_by_id:
                continue
            value = 1.0 - float(distance)
            scores[key] = max(0.0, value)
        return scores
