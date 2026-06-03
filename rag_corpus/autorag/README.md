# AutoRAG data

AutoRAG datasets are exported from:

    rag_corpus/runtime/guidance_chunks.jsonl

Runtime vector retrieval for ViRAGE is built separately with:

    python scripts/rag_corpus/build_runtime_chroma_index.py --chunks rag_corpus/runtime/guidance_chunks.jsonl --persist-dir resources/chroma/virage_guidance_chunks_mxbai_embed_large_latest --collection virage_guidance_chunks_mxbai_embed_large_latest --embedding-provider ollama --embedding-model mxbai-embed-large:latest --embedding-base-url http://localhost:11434 --recreate
