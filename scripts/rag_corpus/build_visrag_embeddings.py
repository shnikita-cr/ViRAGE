from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "src").exists()), Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visrag_core.embeddings import build_embedding_model

DEFAULT_MAX_INPUT_CHARS = 1600


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


def _write_embedding_input_length_report(rows: list[dict], output: Path, max_input_chars: int) -> dict[str, object]:
    items = []
    for row in rows:
        text = vector_text(row)
        items.append({
            "chunk_id": str(row.get("chunk_id") or ""),
            "source_id": str(row.get("source_id") or ""),
            "source_path": str(row.get("source_path") or ""),
            "title": str(row.get("title") or ""),
            "input_char_length": len(text),
            "exceeds_limit": bool(max_input_chars and len(text) > max_input_chars),
        })
    longest = sorted(items, key=lambda item: int(item["input_char_length"]), reverse=True)[:20]
    exceeded = [item for item in items if item["exceeds_limit"]]
    report = {
        "max_input_chars": max_input_chars,
        "total_chunks": len(rows),
        "max_input_char_length": max((int(item["input_char_length"]) for item in items), default=0),
        "exceeded_chunks": len(exceeded),
        "longest_inputs": longest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build precomputed embeddings for VisRAG guidance chunks.")
    parser.add_argument("--input", default="rag_corpus/runtime/guidance_chunks.jsonl")
    parser.add_argument("--output", default="rag_corpus/runtime/guidance_chunk_embeddings.jsonl")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="nomic-embed-text:latest")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-chars", type=int, default=DEFAULT_MAX_INPUT_CHARS)
    parser.add_argument("--input-length-report", default="rag_corpus/runtime/embedding_input_length_report.json")
    args = parser.parse_args()
    rows = read_chunks(ROOT / args.input)
    length_report = _write_embedding_input_length_report(rows, ROOT / args.input_length_report, args.max_input_chars)
    if args.max_input_chars and length_report["exceeded_chunks"]:
        longest = length_report["longest_inputs"][0]
        raise RuntimeError(
            "Embedding input length exceeds configured limit. "
            f"chunk_id={longest['chunk_id']}, "
            f"input_char_length={longest['input_char_length']}, "
            f"max_input_chars={args.max_input_chars}. "
            "Rebuild guidance chunks with a smaller --max-chars value."
        )
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
    print(json.dumps({"input": args.input, "output": args.output, "chunks": len(rows), "input_length_report": args.input_length_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
