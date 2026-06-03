# Runtime VisRAG corpus

Runtime VisRAG uses one text corpus and one Chroma index.

Pipeline:

    python scripts/rag_corpus/export_guidance_chunks.py --min-chars 220 --max-chars 1000 --overlap-chars 120
    python scripts/rag_corpus/build_runtime_chroma_index.py --chunks rag_corpus/runtime/guidance_chunks.jsonl --persist-dir resources/chroma/virage_guidance_chunks_mxbai_embed_large_latest --collection virage_guidance_chunks_mxbai_embed_large_latest --embedding-provider ollama --embedding-model mxbai-embed-large:latest --embedding-base-url http://localhost:11434 --recreate

Runtime retrieval modes:

- `semantic`: Chroma vector retrieval.
- `hybrid`: Chroma vector retrieval plus lexical scoring and metadata weighting.
- `lexical`: BM25 over `rag_corpus/runtime/guidance_chunks.jsonl`.

For `semantic` and `hybrid`, the Chroma manifest must match the current corpus, collection name and embedding model.
