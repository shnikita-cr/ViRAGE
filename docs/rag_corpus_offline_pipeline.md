# Offline RAG Corpus Pipeline

This document describes the new ViRAGE corpus pipeline for rule/guidance retrieval.

The runtime RAG corpus must not contain ready Vega-Lite specifications. ViRAGE retrieves guidance records; `ChartGeneratorService` generates the final Vega-Lite specification.

## Corpus layers

    rag_corpus/raw
      source datasets and raw feedback

    rag_corpus/extracted
      unified source records extracted from raw files

    rag_corpus/processed
      LLM-normalized rule records

    rag_corpus/autorag/virage_rules
      AutoRAG corpus.parquet and qa.parquet

    rag_corpus/runtime
      compact runtime export for future rule retrieval

## Rule types

- `chart_pattern`: which chart family fits a task.
- `readability_rule`: how to keep the chart readable.
- `scale_plot_area_rule`: how to keep the main data pattern visible and avoid outlier-compressed plots.
- `vlm_readability_rule`: what must be visible in static PNG for VLM judge and VLM analysis.

## Full preparation with Ollama

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b

## Full preparation with OpenAI

    python scripts/rag_corpus/run_prepare_corpus.py --provider openai --model gpt-4.1-mini

## Resume and retry

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --resume
    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --retry-failed

## Export for AutoRAG

    python scripts/rag_corpus/run_export_autorag.py

## Export for ViRAGE runtime

    python scripts/rag_corpus/run_export_runtime.py

## Notes

`run_prepare_corpus.py` uses LLM normalization by design. Source extractors only create raw source records; they do not create final processed records.
