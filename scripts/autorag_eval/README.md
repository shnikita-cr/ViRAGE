# ViRAGE AutoRAG evaluation

This directory contains only ViRAGE-to-AutoRAG adapters:

- export the current ViRAGE runtime corpus and labelled retrieval queries to AutoRAG `corpus.parquet` and `qa.parquet`;
- write an AutoRAG YAML config;
- collect AutoRAG `summary.csv` files into a project report.

The retrieval metrics and retriever selection must be computed by AutoRAG through `autorag evaluate`.

## Storage layout

All AutoRAG data, configs, labelled QA files, generated datasets, trials and reports must be stored under:

    rag_corpus/autorag

Current layout:

    rag_corpus/autorag/configs/virage_retrieval_eval.yaml
    rag_corpus/autorag/configs/virage_retrieval_eval.manifest.json
    rag_corpus/autorag/qa/retrieval_queries.jsonl
    rag_corpus/autorag/datasets/runtime/
    rag_corpus/autorag/trials/
    rag_corpus/autorag/report/

Do not put AutoRAG configs or labelled QA files into `configs/autorag` or `data/autorag_eval`.

## Current runtime corpus

Use only the current runtime corpus:

    rag_corpus/runtime/guidance_chunks.jsonl

The historical corpus below must not be used for current evaluation:

    rag_corpus/runtime_pre/virage_rules.jsonl

The file below is used by ViRAGE runtime semantic retrieval, but it is not passed directly into AutoRAG:

    rag_corpus/runtime/guidance_chunk_embeddings.jsonl

AutoRAG semantic retrieval builds its own vector DB from `corpus.parquet` according to the `vectordb` section in the YAML config.

## Embedding models inside AutoRAG

The default AutoRAG config uses Ollama embedding models from the local model list:

    bge-m3:latest
    mxbai-embed-large:latest
    nomic-embed-text:latest
    qwen3-embedding:latest

The config includes lexical BM25 retrieval, semantic retrieval for each listed embedding model, and hybrid retrieval. AutoRAG uses embedding_batch=1 to avoid Ollama batch-level NaN failures. The model embeddinggemma:latest is intentionally excluded from AutoRAG evaluation.

## 1. Export AutoRAG dataset

    python scripts/autorag_eval/export_autorag_dataset.py --corpus rag_corpus/runtime/guidance_chunks.jsonl --queries rag_corpus/autorag/qa/retrieval_queries.jsonl --output-dir rag_corpus/autorag/datasets/runtime

## 2. Build AutoRAG config

Default config includes BM25, semantic VectorDB retrieval with Ollama embeddings, and hybrid retrieval:

    python scripts/autorag_eval/build_autorag_config.py --output rag_corpus/autorag/configs/virage_retrieval_eval.yaml

Lexical-only baseline config:

    python scripts/autorag_eval/build_autorag_config.py --output rag_corpus/autorag/configs/virage_retrieval_eval_lexical.yaml --lexical-only

Custom embedding subset:

    python scripts/autorag_eval/build_autorag_config.py --output rag_corpus/autorag/configs/virage_retrieval_eval.yaml --embedding-models bge-m3:latest,nomic-embed-text:latest

## 3. Run AutoRAG

    autorag evaluate --config rag_corpus/autorag/configs/virage_retrieval_eval.yaml --qa_data_path rag_corpus/autorag/datasets/runtime/qa.parquet --corpus_data_path rag_corpus/autorag/datasets/runtime/corpus.parquet --project_dir rag_corpus/autorag/trials

## 4. Collect AutoRAG results

    python scripts/autorag_eval/collect_autorag_results.py --project-dir rag_corpus/autorag/trials --output-dir rag_corpus/autorag/report

## Output

    rag_corpus/autorag/datasets/runtime/qa.parquet
    rag_corpus/autorag/datasets/runtime/corpus.parquet
    rag_corpus/autorag/datasets/runtime/dataset_manifest.json
    rag_corpus/autorag/trials/**/summary.csv
    rag_corpus/autorag/report/autorag_summary.csv
    rag_corpus/autorag/report/autorag_report.md

## Obsolete files from older layouts

If older patches were applied, delete these files:

    configs/autorag/virage_retrieval_eval.yaml
    configs/autorag/virage_retrieval_eval.manifest.json
    data/autorag_eval/retrieval_queries.jsonl
    src/evaluation/rag_metrics.py
    scripts/rag_eval/run_retrieval_eval.py
    scripts/rag_eval/run_corpus_ablation.py
    scripts/rag_eval/README.md
    data/rag_eval/retrieval_queries.jsonl
    tests/unit/test_rag_eval_metrics.py
    tests/unit/test_rag_eval_scripts.py
