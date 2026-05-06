# ViRAGE RAG corpus workspace

This folder stores only offline RAG assets and scripts.

## Runtime corpus

`rag_corpus/data/` is read by `src.visrag_core` at runtime. It must contain prepared JSON/JSONL rows with the fields expected by `VisRAGCorpus`:

- `id`
- `instruction`
- `chart_type`
- `field_roles`
- `spec_template`

## Iteration 1 AutoRAG assets

`rag_corpus/normalized/jsonl/` contains normalized source corpora for benchmark/export.
`rag_corpus/eval/` contains QA benchmark files.
`rag_corpus/autorag/main/` is the default AutoRAG export target.

## Commands

Validate corpus and QA:

```bash
python rag_corpus/scripts/validate_visrag_corpus.py \
  --corpus-root rag_corpus/normalized/jsonl \
  --qa-root rag_corpus/eval \
  --out-dir rag_corpus/reports \
  --strict
```

Export AutoRAG parquet:

```bash
python rag_corpus/scripts/export_autorag_dataset.py \
  --corpus-src rag_corpus/normalized/jsonl \
  --qa-src rag_corpus/eval \
  --out-dir rag_corpus/autorag/main \
  --strict
```

Parquet export requires `pandas` and either `pyarrow` or `fastparquet`.
