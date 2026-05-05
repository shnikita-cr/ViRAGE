# AutoRAG для Vega-Lite корпуса

Документ фиксирует, как готовить и запускать AutoRAG-оценку для `vega_lite` корпуса в проекте ViRAGE/VisRAG.

## 1. Роль AutoRAG в проекте

AutoRAG используется как offline-инструмент подбора retrieval pipeline для корпуса визуализационных примеров.

Он не является runtime-хранилищем ViRAGE. Runtime VisRAG продолжает читать:

```text
rag_corpus/data/vega_lite_examples.jsonl
```

AutoRAG читает производные parquet-артефакты:

```text
rag_corpus/autorag/vega_lite/corpus.parquet
rag_corpus/autorag/vega_lite/qa_*.parquet
```

## 2. Зачем нужен отдельный AutoRAG layer

AutoRAG помогает сравнить retrieval-варианты:

```text
BM25 tokenizers
BM25 top_k
semantic vector retrieval
hybrid retrieval
query modes
```

Задача AutoRAG:

```text
query → relevant doc_id
```

Задача runtime VisRAG шире:

```text
query + data_profile
→ candidate spec
→ field grounding
→ materialized Vega-Lite spec
→ validator/render pipeline
```

Поэтому AutoRAG-метрики — это не финальная оценка ViRAGE, а offline-сигнал для выбора retrieval-компонентов.

## 3. Входные файлы

### Source of truth

```text
rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl
```

### AutoRAG corpus

```text
rag_corpus/autorag/vega_lite/corpus.parquet
```

Ожидаемые поля:

```text
doc_id
contents
path
metadata
```

Где:

```text
doc_id   = normalized.id
contents = normalized.retrieval_text
path     = normalized.source_path
metadata = source/title/chart_pattern/mark_type/field_roles/etc.
```

### AutoRAG QA

```text
rag_corpus/autorag/vega_lite/qa_instruction.parquet
rag_corpus/autorag/vega_lite/qa_title_query.parquet
rag_corpus/autorag/vega_lite/qa_chart_pattern.parquet
rag_corpus/autorag/vega_lite/qa_all.parquet
rag_corpus/autorag/vega_lite/qa_mixed_technical.parquet
```

## 4. QA-наборы

### `qa_instruction.parquet`

```text
query = instruction
retrieval_gt = exact doc_id
```

Использование:

```text
baseline semantic-ish query без title leakage
```

### `qa_title_query.parquet`

```text
query = title + instruction
retrieval_gt = exact doc_id
```

Использование:

```text
technical smoke: проверяет, что title/file_name индексируется
```

Важно:

```text
это не честный user benchmark, потому что title сильно подсказывает doc_id.
```

### `qa_chart_pattern.parquet`

```text
query = chart_pattern + mark_type + field_roles
retrieval_gt = exact doc_id
```

Использование:

```text
проверка technical retrieval по chart pattern / field roles
```

### `qa_all.parquet`

```text
query = title + instruction + chart_pattern + mark_type + field_roles
```

Использование:

```text
max-information technical retrieval
```

### `qa_mixed_technical.parquet`

Содержит несколько query modes сразу:

```text
query_field
title_query
chart_pattern
```

Использование:

```text
сбалансированный технический набор для сравнения retriever-ов
```

## 5. Порядок подготовки AutoRAG данных

Из корня проекта:

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\00_prepare_qa_sets.ps1
```

Скрипт создаёт/обновляет:

```text
corpus.parquet
qa_instruction.parquet
qa_title_query.parquet
qa_chart_pattern.parquet
qa_all.parquet
qa_mixed_technical.parquet
```

## 6. Retrieval configs

Конфиги лежат здесь:

```text
rag_corpus/autorag/vega_lite/configs/
```

Текущие конфиги:

```text
01_bm25_topk_03.yaml
02_bm25_topk_05.yaml
03_bm25_topk_10.yaml
04_bm25_topk_20.yaml
05_bm25_tokenizer_sweep_topk_05.yaml
06_bm25_tokenizer_sweep_topk_10.yaml
07_semantic_vectordb_default_topk_05.yaml
08_semantic_vectordb_default_topk_10.yaml
09_hybrid_rrf_topk_05.yaml
10_hybrid_cc_topk_05.yaml
11_retrieval_full_grid.yaml
```

## 7. Метрики

Используемые retrieval metrics:

```text
retrieval_f1
retrieval_recall
retrieval_precision
retrieval_ndcg
retrieval_mrr
retrieval_map
```

Если текущая установленная версия AutoRAG не поддерживает `retrieval_map`, убери её из `metrics` в YAML или используй конфиг без `retrieval_map`.

Рекомендуемая интерпретация:

| Metric | Что показывает |
|---|---|
| `retrieval_recall` | попал ли правильный doc_id в top-k |
| `retrieval_precision` | доля правильных документов среди найденных |
| `retrieval_f1` | баланс precision/recall |
| `retrieval_mrr` | насколько высоко стоит первый правильный документ |
| `retrieval_ndcg` | качество ранжирования с учётом позиции |
| `retrieval_map` | средняя precision по релевантным позициям |

Для текущих exact-doc QA наиболее важны:

```text
retrieval_recall
retrieval_mrr
retrieval_ndcg
```

## 8. Запуск BM25 grid

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\01_run_retrieval_grid.ps1
```

По умолчанию запускаются только lexical BM25 configs:

```text
01_bm25_topk_03.yaml
02_bm25_topk_05.yaml
03_bm25_topk_10.yaml
04_bm25_topk_20.yaml
05_bm25_tokenizer_sweep_topk_05.yaml
06_bm25_tokenizer_sweep_topk_10.yaml
```

## 9. Запуск semantic/hybrid configs

Semantic/hybrid retrieval может требовать Chroma/vector dependencies и embedding settings.

Запуск:

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\01_run_retrieval_grid.ps1 -IncludeSemanticHybrid
```

Если semantic/hybrid configs падают из-за окружения, это не блокирует BM25 baseline. Оставь результаты BM25 и вернись к semantic/hybrid после настройки embedding/vector DB.

## 10. Сбор результатов

После запусков:

```powershell
python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py
```

Скрипт создаёт:

```text
rag_corpus/autorag/vega_lite/reports/retrieval_comparison.csv
rag_corpus/autorag/vega_lite/reports/retrieval_comparison.md
```

## 11. Где лежат trials

Запуски складываются в:

```text
rag_corpus/autorag/vega_lite/trials/<qa_name>/<config_name>/
```

Например:

```text
rag_corpus/autorag/vega_lite/trials/qa_mixed_technical/02_bm25_topk_05/
```

## 12. Рекомендуемая последовательность

### Шаг A. Подготовить данные

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\00_prepare_qa_sets.ps1
```

### Шаг B. Запустить BM25 grid

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\01_run_retrieval_grid.ps1
```

### Шаг C. Собрать результаты

```powershell
python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py
```

### Шаг D. Если BM25 baseline стабилен, запустить semantic/hybrid

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\01_run_retrieval_grid.ps1 -IncludeSemanticHybrid
python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py
```

## 13. Что считать хорошим результатом

Для technical QA:

```text
qa_title_query:
  expected recall ≈ 1.0
  это smoke, не честный benchmark

qa_instruction:
  expected ниже, потому что instructions часто похожи

qa_mixed_technical:
  основной технический набор для сравнения retriever-ов
```

Для выбора retriever-а смотреть:

```text
1. qa_mixed_technical retrieval_mrr
2. qa_mixed_technical retrieval_ndcg
3. qa_mixed_technical retrieval_recall
4. qa_instruction retrieval_recall / mrr
5. execution_time
```

## 14. Что не решает AutoRAG

AutoRAG не проверяет:

```text
field grounding по DataProfile
валидность materialized Vega-Lite spec
rendering
scenegraph
empty chart
intent-level correctness
```

Это проверяется pytest integration suite:

```text
tests/integration/test_visrag_*_real_corpus.py
```

## 15. Текущий принцип

```text
AutoRAG выбирает retrieval strategy.
ViRAGE tests проверяют runtime correctness.
```

Не переносить AutoRAG parquet в runtime напрямую.
