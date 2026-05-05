# AutoRAG Vega-Lite

Рабочая директория AutoRAG для `vega_lite` корпуса.

## Основной запуск

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\00_prepare_qa_sets.ps1
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\01_run_retrieval_grid.ps1 -ContinueOnError
python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py
```

Основной config:

```text
rag_corpus/autorag/vega_lite/configs/01_retrieval_grid_all.yaml
```

Он содержит весь retrieval grid в одном YAML:

```text
BM25 top_k 3/5/10/20
Semantic VectorDB top_k 3/5/10/20
Hybrid RRF/CC top_k 3/5/10/20
```

## Debug BM25-only

Если semantic/vector/hybrid окружение ещё не готово:

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\01_run_retrieval_grid.ps1 -DebugBm25Only
python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py
```

## QA sets

По умолчанию прогоняются:

```text
qa_instruction.parquet
qa_title_query.parquet
qa_chart_pattern.parquet
qa_all.parquet
qa_mixed_technical.parquet
```

## Reports

```text
rag_corpus/autorag/vega_lite/reports/retrieval_comparison.csv
rag_corpus/autorag/vega_lite/reports/retrieval_comparison.md
rag_corpus/autorag/vega_lite/reports/retrieval_comparison.json
```

Подробнее см. `docs/autorag_vega_lite.md`.
