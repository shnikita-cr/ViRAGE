# VisRAG Corpus Pipeline

Документ фиксирует текущий pipeline подготовки корпуса для VisRAG в проекте ViRAGE, разделение ролей между JSONL и Parquet, порядок воспроизведения, тесты и текущие ограничения.

## 1. Назначение pipeline

Цель pipeline — подготовить корпус визуализационных примеров так, чтобы он мог использоваться в двух режимах:

1. **Offline evaluation / optimization** через AutoRAG.
2. **Runtime integration** внутри ViRAGE VisRAG-сервиса.

Текущий подтверждённый путь:

```text
cleaned corpus
→ normalized JSONL
→ AutoRAG corpus/QA parquet
→ ViRAGE runtime JSONL
→ VisRAGCorpus.load()
→ VisRAGCoreService.search()
→ VisRAGService
→ ChartGeneratorService
→ SpecValidatorService
→ VegaLitePlotDrawingService
→ ScenegraphCheckService
→ EmptyChartCheckService
→ pytest integration tests
```

Текущий MVP-корпус: `vega_lite`.

---

## 2. Роли директорий

### `rag_corpus/raw/`

Сырой backup-слой.

Используется для хранения исходных репозиториев и датасетов. Не является рабочим runtime-источником.

```text
rag_corpus/raw/
```

Правило:

```text
raw не используется напрямую в VisRAG runtime.
raw можно пересоздать загрузочными скриптами.
raw может быть в .gitignore или храниться как локальный backup.
```

---

### `rag_corpus/cleaned/`

Ручной curated-слой.

Здесь лежат только нужные или потенциально нужные файлы после ручной чистки.

```text
rag_corpus/cleaned/
```

Назначение:

```text
cleaned = вход для нормализаторов.
```

---

### `rag_corpus/normalized/jsonl/`

Source of truth после нормализации.

```text
rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl
```

Этот файл содержит полный normalized record:

```json
{
  "id": "vega_lite:example:point_2d",
  "source": "vega-lite",
  "source_path": "vega-lite/examples/specs/point_2d.vl.json",
  "file_name": "point_2d.vl.json",
  "file_stem": "point_2d",
  "title": "point 2d",
  "corpus_type": "example",
  "instruction": "Create a scatter plot...",
  "mark_type": "point",
  "chart_pattern": "scatter_plot",
  "field_roles": {
    "x": "quantitative",
    "y": "quantitative"
  },
  "retrieval_text": "Title: point 2d. Instruction: ...",
  "spec_template": {},
  "raw_spec": {},
  "data_policy": "removed_from_spec_template"
}
```

Правило:

```text
normalized JSONL = главный source of truth.
```

---

### `rag_corpus/autorag/`

Производные файлы для AutoRAG.

```text
rag_corpus/autorag/vega_lite/corpus.parquet
rag_corpus/autorag/vega_lite/qa.parquet
```

Назначение:

```text
AutoRAG offline retrieval evaluation / optimization.
```

`corpus.parquet` содержит:

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
metadata = короткие признаки: source, title, chart_pattern, mark_type, field_roles...
```

`qa.parquet` содержит:

```text
qid
query
retrieval_gt
generation_gt
metadata
```

Важно:

```text
AutoRAG parquet-файлы не являются runtime-корпусом ViRAGE.
```

---

### `rag_corpus/data/`

Runtime corpus для текущего `src/visrag_core`.

```text
rag_corpus/data/vega_lite_examples.jsonl
```

Это адаптированная версия normalized JSONL под runtime-контракт VisRAG:

```json
{
  "id": "vega_lite:example:point_2d",
  "source": "vega-lite",
  "corpus": "vega_lite",
  "corpus_type": "example",
  "instruction": "Title: point 2d. Instruction: ...",
  "description": "Create a scatter plot...",
  "keywords": ["point 2d", "point_2d", "scatter_plot", "point"],
  "chart_type": "point",
  "field_roles": {
    "x": "quantitative",
    "y": "quantitative"
  },
  "transform_types": [],
  "spec_template": {},
  "metadata": {}
}
```

Правило:

```text
rag_corpus/data/*.jsonl читается runtime-кодом ViRAGE.
```

---

## 3. Почему JSONL и Parquet разделены

### JSONL

Используется для:

```text
normalized source of truth
runtime corpus
payload/spec_template storage
ручной проверки
git diff
debug trace
```

Преимущества:

```text
читается человеком
удобен для diff
удобен для ручной диагностики
уже поддержан src/visrag_core/corpus.py
```

---

### Parquet

Используется для:

```text
AutoRAG corpus dataset
AutoRAG QA dataset
offline retrieval evaluation
```

Преимущества:

```text
ожидается AutoRAG
удобен для табличных метрик
быстро читается pandas/pyarrow
```

---

## 4. Порядок скриптов

Скрипты расположены в порядке воспроизведения:

```text
scripts/rag_corpus/
  00_load_raw_repositories.ps1
  01_load_chart_llm_hf.ps1
  02_generate_raw_manifest.ps1
  03_append_chart_llm_hf_manifest.ps1
  04_inspect_raw_corpus.py
  05_inspect_cleaned_corpus.py
  06_normalize_vega_lite_examples.py
  07_export_autorag_corpus.py
  08_export_autorag_qa_from_normalized.py
  09_export_visrag_runtime_corpus.py
  10_run_pipeline_from_cleaned.ps1
  README.md
```

---

## 5. Быстрый запуск pipeline от cleaned

Основной воспроизводимый запуск:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rag_corpus\10_run_pipeline_from_cleaned.ps1
```

Ожидаемые артефакты:

```text
rag_corpus/reports/cleaned_inventory.md
rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl
rag_corpus/reports/official_vega_lite_examples_report.md
rag_corpus/autorag/vega_lite/corpus.parquet
rag_corpus/autorag/vega_lite/qa.parquet
rag_corpus/data/vega_lite_examples.jsonl
rag_corpus/reports/visrag_runtime_export_report.md
```

---

## 6. Runtime export policy

Скрипт:

```text
scripts/rag_corpus/09_export_visrag_runtime_corpus.py
```

Преобразует:

```text
rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl
```

в:

```text
rag_corpus/data/vega_lite_examples.jsonl
```

### Что делает exporter

```text
1. Удаляет data/datasets.
2. Удаляет чужие Vega-Lite field names.
3. Преобразует chart_pattern / mark_type в runtime chart_type.
4. Приводит field_roles к runtime semantic types.
5. Если field_roles пустые, восстанавливает их из spec_template.encoding.
6. Пропускает сложные layer/facet/repeat/concat specs.
7. Пропускает transform-heavy specs.
8. Пропускает interactive params/select по умолчанию.
9. Удаляет params/selection/condition из runtime spec_template.
10. Пересобирает instruction из очищенных runtime field_roles.
11. Пишет Markdown/JSON report.
```

### Runtime-safe semantic roles

```text
quantitative
temporal
nominal
ordinal
boolean
geojson
```

### Runtime-safe chart types на текущем этапе

```text
point
line
bar
histogram
area
rect
tick
rule
text
circle
square
boxplot
```

---

## 7. Почему `data` не хранится в `spec_template`

В ViRAGE данные подключаются runtime-слоем.

Текущий ChartGenerator вставляет:

```json
{
  "data": {
    "url": "<prepared_dataset_path>"
  }
}
```

Поэтому в корпусе запрещено использовать чужие:

```text
data
datasets
external data URLs
```

Правило:

```text
spec_template = visual structure only.
data.url attaches later by ChartGeneratorService.
```

---

## 8. Pytest integration tests

Smoke-проверки перенесены в pytest.

### Runtime corpus test

```text
tests/integration/test_visrag_runtime_corpus_real.py
```

Проверяет:

```text
VisRAGCorpus.load()
runtime-safe field_roles
отсутствие data/datasets
отсутствие чужих field names
отсутствие params/selection/condition
покрытие базовых chart types
```

---

### Core search test

```text
tests/integration/test_visrag_core_search_real_corpus.py
```

Проверяет:

```text
VisRAGCoreService.search()
candidate retrieval
field_mapping
materialized spec_template
```

Кейсы:

```text
scatter
line
bar
histogram
heatmap xfail
```

---

### Chart pipeline test

```text
tests/integration/test_visrag_chart_pipeline_real_corpus.py
```

Проверяет:

```text
VisRAGService
→ ChartGeneratorService
→ SpecValidatorService
```

Кейсы:

```text
scatter
line
bar
histogram
```

---

### Render pipeline test

```text
tests/integration/test_visrag_render_pipeline_real_corpus.py
```

Проверяет:

```text
VisRAGService
→ ChartGeneratorService
→ SpecValidatorService
→ VegaLitePlotDrawingService
→ ScenegraphCheckService
→ EmptyChartCheckService
```

Кейсы:

```text
scatter
line
bar
histogram
```

---

## 9. Команды проверки

### Все integration tests

```powershell
pytest tests\integration -q
```

### По отдельности

```powershell
pytest tests\integration\test_visrag_runtime_corpus_real.py -q
pytest tests\integration\test_visrag_core_search_real_corpus.py -q
pytest tests\integration\test_visrag_chart_pipeline_real_corpus.py -q
pytest tests\integration\test_visrag_render_pipeline_real_corpus.py -q
```

---

## 10. Текущий зелёный baseline

На текущем этапе подтверждён путь:

```text
cleaned corpus
→ normalized JSONL
→ AutoRAG corpus/QA parquet
→ ViRAGE runtime JSONL
→ VisRAGCorpus.load()
→ VisRAGCoreService.search()
→ VisRAGService
→ ChartGeneratorService
→ SpecValidatorService
→ VegaLitePlotDrawingService
→ ScenegraphCheckService
→ EmptyChartCheckService
→ pytest integration tests
```

Подтверждённые рабочие кейсы:

```text
scatter_quantitative_relationship
line_temporal_trend
bar_category_comparison
histogram_distribution
```

Текущий runtime corpus содержит около 40 included examples из 100 normalized records.

---

## 11. Known issues

### 11.1. `selected_fields` пока preferred, не strict

Сейчас field grounding может использовать поле не из `selected_fields`, если оно лучше подходит по semantic type.

Пример:

```text
query: compare values across categories
selected_fields: Category, Sales
actual mapping: x = Month, y = Sales
```

Это технически валидный spec, но семантически спорный.

Backlog:

```text
selected_fields_policy:
  prefer
  strict
  soft_fail
```

---

### 11.2. Heatmap / rect пока не runtime-stable

Heatmap сейчас отмечен как `xfail`.

Вероятные причины:

```text
мало runtime-safe heatmap examples
field_roles для x/y/color требуют отдельной настройки
часть heatmap examples содержит transforms или complex layout
```

---

### 11.3. Некоторые line templates слишком специфичные

Например:

```text
time_output_utc_scale
```

Может содержать:

```text
timeUnit = yearmonthdatehoursminutes
scale.type = utc
```

Backlog:

```text
ranking penalty за overly-specific templates
```

---

### 11.4. Runtime corpus пока небольшой

Текущий strict exporter оставляет только runtime-safe subset.

Это правильно для MVP, но позже можно расширять покрытие:

```text
support transforms
support layer/facet
support rect/heatmap
support arc/pie, если chart_types расширятся
```

---

## 12. Следующие итерации

### Итерация 1. Intent-level QA для VisRAG

Сделать ручной набор:

```text
query
data_profile
expected_chart_pattern
expected_field_roles
acceptable_doc_ids
```

Цель:

```text
оценивать не exact doc_id, а intent-level correctness.
```

---

### Итерация 2. Grounding policy

Добавить режимы:

```text
prefer
strict
soft_fail
```

Цель:

```text
не подменять выбранные Request Analyzer поля без явного основания.
```

---

### Итерация 3. Heatmap support

Отдельно стабилизировать:

```text
rect / heatmap examples
x nominal
y nominal
color quantitative
```

---

### Итерация 4. Расширение runtime corpus

Подключать новые источники только после зелёного baseline на `vega_lite`.

Кандидаты:

```text
chart-llm-hf
nlvcorpus
local_cases
chart-llm
draco
compassql
```

Правило:

```text
каждый новый corpus должен иметь:
normalized JSONL
AutoRAG export
runtime JSONL export
pytest integration tests
source/leakage policy
```

---

## 13. Правила добавления новых корпусов

Для каждого нового корпуса:

```text
1. Не читать raw напрямую.
2. Сначала привести cleaned.
3. Создать normalized JSONL.
4. Добавить report.
5. Добавить AutoRAG corpus/QA export.
6. Добавить runtime JSONL export.
7. Проверить через pytest.
8. Пометить benchmark-sensitive источники.
```

Особенно важно для:

```text
chart-llm
nlvcorpus
chart-llm-hf
```

Если источник используется в VegaChat benchmark, он должен иметь:

```text
is_eval_leak_sensitive = true
benchmark_group = ...
source_split = ...
```

---

## 14. Definition of Done для текущего MVP

MVP VisRAG corpus integration считается готовым, если:

```text
[done] runtime corpus генерируется
[done] runtime corpus загружается
[done] core search работает
[done] chart pipeline работает
[done] render pipeline работает
[done] pytest integration tests зелёные
[done] known issues зафиксированы
```
