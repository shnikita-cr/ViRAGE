from __future__ import annotations

import argparse
from pathlib import Path

from src.services.visrag_retrieval import retrieve_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or refresh VisRAG semantic indexes from normalized corpora.")
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--embedding-backend", default="local_tfidf")
    parser.add_argument("--embedding-model", default="embeddinggemma")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434/api")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    bundle = retrieve_examples(
        Path(args.corpus_root),
        index_root=Path(args.index_root),
        query_text="bootstrap semantic index for visualization retrieval",
        preferred_chart_types=["line", "bar", "scatter", "histogram", "boxplot"],
        top_k=1,
        fetch_k=2,
        similarity_threshold=0.0,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        ollama_base_url=args.ollama_base_url,
        ollama_timeout_seconds=30.0,
        force_rebuild=args.force,
    )
    print("Backend:", bundle.backend_name)
    for corpus, status in bundle.corpus_status.items():
        print(f"{corpus}: {status}")


if __name__ == "__main__":
    main()
