# AutoRAG export

Use `rag_corpus/scripts/export_autorag_dataset.py` to build AutoRAG parquet datasets from two separate folders:

```bash
python rag_corpus/scripts/export_autorag_dataset.py \
  --corpus-src rag_corpus/data \
  --qa-src rag_corpus/eval \
  --out-dir rag_corpus/autorag/out
```

The script writes `corpus.parquet` and `qa.parquet`.
