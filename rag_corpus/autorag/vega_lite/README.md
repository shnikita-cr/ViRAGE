# AutoRAG Vega-Lite

Рабочая директория AutoRAG для `vega_lite` корпуса.

## Быстрый запуск

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\00_prepare_qa_sets.ps1
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\01_run_retrieval_grid.ps1
python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py
```

## Semantic / hybrid retrieval

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\01_run_retrieval_grid.ps1 -IncludeSemanticHybrid -ContinueOnError
python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py
```

Подробнее см. `docs/autorag_vega_lite.md`.
