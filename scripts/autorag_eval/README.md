# AutoRAG evaluation export

Export AutoRAG-compatible parquet files from the runtime guidance corpus:

    python scripts/autorag_eval/export_autorag_dataset.py --corpus rag_corpus/runtime/guidance_chunks.jsonl --queries rag_corpus/autorag/qa/retrieval_queries.jsonl --output-dir rag_corpus/autorag/datasets/runtime

The ViRAGE runtime index is built with `scripts/rag_corpus/build_runtime_chroma_index.py`.
