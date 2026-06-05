from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in _CURRENT_FILE.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    Path.cwd(),
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visrag_core.chroma_index import (  # noqa: E402
    chunk_embedding_text,
    normalize_chroma_metadata,
    validate_chunks_for_index,
    write_chroma_manifest,
)
from src.visrag_core.embeddings import build_embedding_model  # noqa: E402
from src.visrag_core.stores.jsonl_store import JsonlVisRAGStore  # noqa: E402


def _batched(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _embed_documents(embedder: Any, texts: list[str]) -> list[list[float]]:
    if hasattr(embedder, "embed_documents"):
        return [list(vector) for vector in embedder.embed_documents(texts)]
    return [list(embedder.embed_query(text)) for text in texts]


def build_runtime_chroma_index(
        *,
        chunks_path: Path,
        persist_dir: Path,
        collection_name: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_base_url: str | None,
        batch_size: int,
        max_input_chars: int,
        recreate: bool,
) -> dict[str, Any]:
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Guidance chunks file not found: {chunks_path}")
    if recreate and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    store = JsonlVisRAGStore(chunks_path)
    chunks = store.load_chunks()
    validate_chunks_for_index(chunks)

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Chroma index build requires the chromadb package.") from exc

    embedder = build_embedding_model(
        provider=embedding_provider,
        model=embedding_model,
        base_url=embedding_base_url,
    )
    client = chromadb.PersistentClient(path=str(persist_dir))
    existing = {getattr(item, "name", str(item)) for item in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    embedding_dimension: int | None = None
    total = 0
    batches = _batched(chunks, max(1, int(batch_size)))
    for batch in batches:
        ids = [chunk.chunk_id for chunk in batch]
        documents = [chunk_embedding_text(chunk, max_chars=max_input_chars) for chunk in batch]
        vectors = _embed_documents(embedder, documents)
        if len(vectors) != len(batch):
            raise RuntimeError(f"Embedding model returned {len(vectors)} vectors for {len(batch)} documents.")
        for chunk, vector in zip(batch, vectors, strict=True):
            if not vector:
                raise RuntimeError(f"Embedding model returned an empty vector for chunk_id={chunk.chunk_id!r}.")
            if embedding_dimension is None:
                embedding_dimension = len(vector)
            elif len(vector) != embedding_dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch for chunk_id={chunk.chunk_id!r}: "
                    f"expected {embedding_dimension}, got {len(vector)}."
                )
        metadatas = [
            normalize_chroma_metadata(
                {
                    **(chunk.metadata or {}),
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "source_name": chunk.source_name,
                    "source_kind": chunk.source_kind,
                    "title": chunk.title,
                    "source_path": chunk.source_path or "",
                    "url": chunk.url or "",
                }
            )
            for chunk in batch
        ]
        collection.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
        total += len(batch)
        print(f"indexed {total}/{len(chunks)} chunks", flush=True)

    if embedding_dimension is None:
        raise RuntimeError("No embeddings were produced for the runtime corpus.")
    manifest = write_chroma_manifest(
        persist_dir=persist_dir,
        chunks_path=chunks_path,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        chunk_count=len(chunks),
    )
    return {"persist_dir": persist_dir.as_posix(), "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ViRAGE runtime Chroma index from guidance chunks.")
    parser.add_argument("--chunks", default="rag_corpus/runtime/guidance_chunks.jsonl")
    parser.add_argument("--persist-dir", default="resources/chroma/virage_guidance_chunks_nomic_embed_text_latest")
    parser.add_argument("--collection", default="virage_guidance_chunks_nomic_embed_text_latest")
    parser.add_argument("--embedding-provider", default="ollama")
    parser.add_argument("--embedding-model", default="nomic-embed-text:latest")
    parser.add_argument("--embedding-base-url", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-chars", type=int, default=1600)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_runtime_chroma_index(
        chunks_path=Path(args.chunks),
        persist_dir=Path(args.persist_dir),
        collection_name=str(args.collection),
        embedding_provider=str(args.embedding_provider),
        embedding_model=str(args.embedding_model),
        embedding_base_url=str(args.embedding_base_url or "") or None,
        batch_size=int(args.batch_size),
        max_input_chars=int(args.max_input_chars),
        recreate=bool(args.recreate),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
