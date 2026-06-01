# ViRAGE AutoRAG workspace

All AutoRAG-related data, configs, labelled QA files, generated datasets, trial outputs and reports must live in this directory.

## Required files

    rag_corpus/autorag/configs/virage_retrieval_eval.yaml
    rag_corpus/autorag/qa/retrieval_queries.jsonl

## Generated files

    rag_corpus/autorag/datasets/runtime/corpus.parquet
    rag_corpus/autorag/datasets/runtime/qa.parquet
    rag_corpus/autorag/trials/
    rag_corpus/autorag/report/

## Current corpus

The current ViRAGE runtime corpus is outside this directory:

    rag_corpus/runtime/guidance_chunks.jsonl

The historical corpus below must not be used for current AutoRAG evaluation:

    rag_corpus/runtime_pre/virage_rules.jsonl

## Runtime embeddings

The file below belongs to ViRAGE runtime retrieval and is not passed directly to AutoRAG:

    rag_corpus/runtime/guidance_chunk_embeddings.jsonl

AutoRAG builds vector stores from `corpus.parquet` using the embedding models declared in `configs/virage_retrieval_eval.yaml`.

## AutoRAG embedding policy

AutoRAG uses only these Ollama embedding models:

    bge-m3:latest
    mxbai-embed-large:latest
    nomic-embed-text:latest
    qwen3-embedding:latest

The model below is intentionally excluded from AutoRAG evaluation:

    embeddinggemma:latest

The generated AutoRAG config uses:

    embedding_batch: 1

This keeps semantic and hybrid retrieval more stable with Ollama API embeddings.
