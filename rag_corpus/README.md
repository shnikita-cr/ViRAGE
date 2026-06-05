# ViRAGE RAG corpus

The runtime corpus is:

    rag_corpus/runtime/guidance_chunks.jsonl

The runtime index is built before running ViRAGE:

    python scripts/rag_corpus/build_runtime_chroma_index.py --chunks rag_corpus/runtime/guidance_chunks.jsonl --persist-dir resources/chroma/virage_guidance_chunks_nomic_embed_text_latest --collection virage_guidance_chunks_nomic_embed_text_latest --embedding-provider ollama --embedding-model nomic-embed-text:latest --embedding-base-url http://localhost:11434 --recreate

Feedback is collected as raw records under `rag_corpus/feedback`, normalized during corpus export, and then included in `guidance_chunks.jsonl`.

Supported runtime retrieval modes:

- `semantic`
- `hybrid`
- `lexical`
