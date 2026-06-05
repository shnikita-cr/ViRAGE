# ViRAGE runtime RAG howto

1. Export the runtime corpus.

    python scripts/rag_corpus/export_guidance_chunks.py --min-chars 220 --max-chars 1000 --overlap-chars 120

2. Build the runtime Chroma index.

    python scripts/rag_corpus/build_runtime_chroma_index.py --chunks rag_corpus/runtime/guidance_chunks.jsonl --persist-dir resources/chroma/virage_guidance_chunks_nomic_embed_text_latest --collection virage_guidance_chunks_nomic_embed_text_latest --embedding-provider ollama --embedding-model nomic-embed-text:latest --embedding-base-url http://localhost:11434 --batch-size 8 --max-input-chars 1600 --recreate

3. Run a small E2E benchmark.

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --run-id e2e_cloud_chroma_small --max-cases 5 --execute

Runtime retrieval modes:

- `semantic`: vector retrieval from the Chroma index.
- `hybrid`: vector retrieval from Chroma combined with lexical scoring and metadata weighting.
- `lexical`: BM25 over the current runtime corpus.

After changing `rag_corpus/runtime/guidance_chunks.jsonl`, rebuild the Chroma index before running `semantic` or `hybrid`.
