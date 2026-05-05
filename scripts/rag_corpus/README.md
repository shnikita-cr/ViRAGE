# RAG corpus scripts

The scripts are numbered to make the corpus pipeline repeatable.

## Raw source bootstrap

These steps are optional after `rag_corpus/raw` is already available.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rag_corpus\00_load_raw_repositories.ps1
powershell -ExecutionPolicy Bypass -File scripts\rag_corpus\01_load_chart_llm_hf.ps1
powershell -ExecutionPolicy Bypass -File scripts\rag_corpus\02_generate_raw_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\rag_corpus\03_append_chart_llm_hf_manifest.ps1
python scripts\rag_corpus\04_inspect_raw_corpus.py --raw-root rag_corpus\raw --out-dir rag_corpus\reports
```

## Current repeatable pipeline from cleaned corpus

Use this after `rag_corpus/cleaned` has been curated manually.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rag_corpus\10_run_pipeline_from_cleaned.ps1
```

This runs:

1. cleaned inventory;
2. Vega-Lite normalization;
3. AutoRAG corpus export;
4. AutoRAG QA export;
5. ViRAGE runtime corpus export;
6. real-corpus VisRAG pytest smoke tests.

## File roles

- `rag_corpus/normalized/jsonl/*.jsonl` is the source-of-truth normalized corpus.
- `rag_corpus/autorag/**/*.parquet` is derived data for AutoRAG offline evaluation.
- `rag_corpus/data/*.jsonl` is the runtime corpus consumed by `src/visrag_core`.
- Smoke checks live in `tests/integration`, not in `scripts/rag_corpus`.
