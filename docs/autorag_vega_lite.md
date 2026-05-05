# AutoRAG Vega-Lite

Документ описывает запуск AutoRAG для подбора retrieval-компонентов на корпусе `vega_lite`.

## 1. Роль AutoRAG в проекте

AutoRAG используется только для offline retrieval evaluation / optimization.

Он не заменяет runtime VisRAG-корпус:

```text
ViRAGE runtime:
  rag_corpus/data/vega_lite_examples.jsonl

AutoRAG offline:
  rag_corpus/autorag/vega_lite/corpus.parquet
  rag_corpus/autorag/vega_lite/qa_*.parquet
```

AutoRAG помогает выбрать retrieval-подход и параметры, но runtime-код проекта продолжает читать JSONL через `src/visrag_core`.

---

## 2. Артефакты

Рабочая директория:

```text
rag_corpus/autorag/vega_lite/
```

Основные файлы:

```text
00_prepare_qa_sets.ps1
01_run_retrieval_grid.ps1
02_collect_retrieval_results.py
README.md
configs/01_retrieval_grid_all.yaml
configs/99_retrieval_grid_bm25_debug.yaml
```

---

## 3. Почему один all-grid config

Раньше были отдельные YAML для разных `top_k` и retrieval-типов.

Теперь основной конфиг один:

```text
configs/01_retrieval_grid_all.yaml
```

Он содержит несколько `node_line` внутри одного YAML.

Почему так:

```text
1. один config проще версионировать;
2. один запуск проще воспроизводить;
3. все результаты оказываются в одном trial tree;
4. top_k остаётся scalar внутри каждого node_line;
5. нет риска несовместимости из-за top_k: [3, 5, 10, 20].
```

AutoRAG документация описывает `top_k` как node-level параметр, а пример retrieval config использует scalar `top_k`. Поэтому grid по `top_k` сделан через отдельные `node_line`, а не через список значений.

---

## 4. Что проверяет all-grid

Основной config сравнивает:

```text
BM25 lexical retrieval:
  top_k = 3, 5, 10, 20
  bm25_tokenizer = porter_stemmer, space

Semantic retrieval:
  module = vectordb
  vectordb = default
  top_k = 3, 5, 10, 20

Hybrid retrieval:
  modules = hybrid_rrf, hybrid_cc
  top_k = 3, 5, 10, 20
```

Hybrid retrieval в AutoRAG требует lexical и semantic retrieval nodes в том же config; поэтому hybrid node lines содержат `lexical_retrieval`, `semantic_retrieval` и `hybrid_retrieval` вместе.

---

## 5. Метрики

Используемые retrieval-метрики:

```text
retrieval_f1
retrieval_recall
retrieval_precision
retrieval_ndcg
retrieval_mrr
```

Интерпретация:

```text
retrieval_recall:
  есть ли правильный документ среди top-k

retrieval_mrr:
  насколько высоко стоит первый правильный документ

retrieval_ndcg:
  насколько хорошо ранжирование в целом

retrieval_precision:
  сколько retrieved документов релевантны

retrieval_f1:
  баланс precision/recall
```

---

## 6. QA-наборы

Скрипт подготовки создаёт:

```text
qa_instruction.parquet
qa_title_query.parquet
qa_chart_pattern.parquet
qa_all.parquet
qa_mixed_technical.parquet
```

Назначение:

```text
qa_instruction:
  базовый semantic-ish smoke по instruction

qa_title_query:
  проверяет индексирование title/file name; не считать честным benchmark

qa_chart_pattern:
  проверяет поиск по chart pattern

qa_all:
  широкий technical set

qa_mixed_technical:
  сбалансированный technical smoke для нескольких query modes
```

---

## 7. Подготовка QA и corpus

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\00_prepare_qa_sets.ps1
```

Создаёт:

```text
rag_corpus/autorag/vega_lite/corpus.parquet
rag_corpus/autorag/vega_lite/qa_instruction.parquet
rag_corpus/autorag/vega_lite/qa_title_query.parquet
rag_corpus/autorag/vega_lite/qa_chart_pattern.parquet
rag_corpus/autorag/vega_lite/qa_all.parquet
rag_corpus/autorag/vega_lite/qa_mixed_technical.parquet
```

---

## 8. Запуск all-grid

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\01_run_retrieval_grid.ps1 -ContinueOnError
```

`-ContinueOnError` рекомендуется, потому что semantic/vector/hybrid могут зависеть от локального embedding/vector-db окружения.

---

## 9. BM25 debug-only запуск

Если нужно проверить только быстрый lexical baseline:

```powershell
powershell -ExecutionPolicy Bypass -File rag_corpus\autorag\vega_lite\01_run_retrieval_grid.ps1 -DebugBm25Only
```

Использует:

```text
configs/99_retrieval_grid_bm25_debug.yaml
```

---

## 10. Сбор результатов

```powershell
python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py
```

Создаёт:

```text
rag_corpus/autorag/vega_lite/reports/retrieval_comparison.csv
rag_corpus/autorag/vega_lite/reports/retrieval_comparison.md
rag_corpus/autorag/vega_lite/reports/retrieval_comparison.json
```

---

## 11. Как читать результаты

Сначала смотреть:

```text
qa_mixed_technical
qa_instruction
```

`qa_title_query` использовать только как smoke того, что title индексируется.

Если:

```text
recall@20 высокий, но mrr/ndcg низкие
```

значит документ находится, но плохо ранжируется.

Если:

```text
recall@20 низкий
```

значит проблема в корпусе, query или retrieval model.

---

## 12. DoD AutoRAG итерации

```text
[ ] corpus.parquet создан
[ ] qa_*.parquet созданы
[ ] 01_retrieval_grid_all.yaml запущен хотя бы на qa_mixed_technical
[ ] retrieval_comparison.md создан
[ ] лучший config определён по mrr/ndcg/recall
[ ] выводы перенесены в VisRAG backlog
```
