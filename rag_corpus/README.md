# ViRAGE RAG Corpus

This directory stores the offline corpus pipeline for ViRAGE rule/guidance retrieval.

The new corpus does **not** store runtime Vega-Lite specifications. Runtime retrieval should return guidance records such as chart patterns, readability rules, scale/plot-area rules, and VLM readability rules. `ChartGeneratorService` remains responsible for producing the final Vega-Lite specification.

## Layout

- `raw/` — source datasets and raw inputs. Do not use directly at runtime.
- `extracted/` — normalized source records extracted from raw files.
- `processed/` — LLM-normalized rule records.
- `autorag/virage_rules/` — AutoRAG parquet exports and reports.
- `runtime/` — compact runtime export for future VisRAG rule retrieval.
- `reports/` — corpus preparation summaries.

## Record types

- `chart_pattern` — which visualization family fits a task.
- `readability_rule` — how to keep the chart readable.
- `scale_plot_area_rule` — how to use plot area well and handle outlier-compressed charts carefully.
- `vlm_readability_rule` — what must be visible in a static PNG for VLM judge/analysis.

## Main commands

Prepare processed records with Ollama:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b

Prepare processed records with OpenAI:

    python scripts/rag_corpus/run_prepare_corpus.py --provider openai --model gpt-4.1-mini

Export AutoRAG files:

    python scripts/rag_corpus/run_export_autorag.py

Export runtime rules:

    python scripts/rag_corpus/run_export_runtime.py

## Important policy

Runtime rule documents must not contain Vega-Lite `mark`, `encoding`, `$schema`, or `spec_template` payloads. They should provide concise generation guidance, not templates for copying.
