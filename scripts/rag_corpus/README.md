# ViRAGE RAG corpus scripts

These scripts prepare offline RAG corpora for ViRAGE. The pipeline is LLM-normalization based and produces rule/guidance documents, not Vega-Lite spec templates.

## Basic flow

    python scripts/rag_corpus/sources/scan_sources.py
    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b
    python scripts/rag_corpus/run_export_autorag.py
    python scripts/rag_corpus/run_export_runtime.py

## With OpenAI

    python scripts/rag_corpus/run_prepare_corpus.py --provider openai --model gpt-4.1-mini

## Resume / retry

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --resume
    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --retry-failed

## Folder roles

- `common/` — shared schemas, IO, LLM client, validation.
- `sources/` — extraction from raw datasets.
- `normalize/` — LLM normalization, merge, dedupe, validation.
- `autorag/` — export to AutoRAG parquet files.
- `runtime/` — export compact runtime rule documents.

## Rule types

- `chart_pattern`
- `readability_rule`
- `scale_plot_area_rule`
- `vlm_readability_rule`
