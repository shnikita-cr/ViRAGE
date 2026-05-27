from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "src").exists()), Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visrag_core.embeddings import build_embedding_model


def read_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Guidance chunks file not found: {path}. Run export_guidance_chunks.py first.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Guidance chunks file is empty: {path}")
    return rows


def vector_text(row: dict) -> str:
    metadata = row.get("metadata") or {}
    return "\n".join([
        str(row.get("title") or ""),
        str(row.get("source_name") or ""),
        str(row.get("source_kind") or ""),
        str(metadata.get("chart_family") or ""),
        str(row.get("text") or ""),
    ]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build precomputed embeddings for VisRAG guidance chunks.")
    parser.add_argument("--input", default="rag_corpus/runtime/guidance_chunks.jsonl")
    parser.add_argument("--output", default="rag_corpus/runtime/guidance_chunk_embeddings.jsonl")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    rows = read_chunks(ROOT / args.input)
    embedder = build_embedding_model(provider=args.provider, model=args.model, base_url=args.base_url or None)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), max(1, args.batch_size)):
            batch = rows[start:start + max(1, args.batch_size)]
            vectors = embedder.embed_documents([vector_text(row) for row in batch])
            for row, vector in zip(batch, vectors):
                handle.write(json.dumps({"chunk_id": row["chunk_id"], "embedding": list(vector)}, ensure_ascii=False) + "\n")
            print(f"embedded {min(start + len(batch), len(rows))}/{len(rows)}")
    print(json.dumps({"input": args.input, "output": args.output, "chunks": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
