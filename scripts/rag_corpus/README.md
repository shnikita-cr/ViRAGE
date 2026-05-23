# ViRAGE RAG corpus scripts

These scripts prepare offline RAG corpora for ViRAGE. The pipeline is LLM-normalization based and produces rule/guidance documents, not Vega-Lite spec templates.

## Basic flow

    python scripts/rag_corpus/sources/scan_sources.py
    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b
    python scripts/rag_corpus/run_export_autorag.py
    python scripts/rag_corpus/run_export_runtime.py

## ChartSquared modes

ChartSquared / C-2 can be large, so it has three extraction modes:

- `prompts_only` — extracts only ChartAF/prompt files. This is the cheapest and most useful mode for VLM judge and feedback rules.
- `sample` — extracts all prompt files plus a deterministic sample of ChartUIE/task files. This is the default.
- `full` — extracts every supported ChartSquared file. Use only for long offline runs.

Examples:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources chartsquared --chartsquared-mode prompts_only

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources chartsquared --chartsquared-mode sample --chartsquared-limit 300

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources chartsquared --chartsquared-mode full

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
- `domain_semantics_rule`
