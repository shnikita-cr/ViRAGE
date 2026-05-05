# VisRAG Corpus Pipeline

Документ фиксирует pipeline подготовки корпуса для VisRAG в проекте ViRAGE, разделение ролей между JSONL и Parquet, порядок воспроизведения, тесты, текущий зелёный baseline и known issues после внедрения `auto/strict` grounding policy.

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

Текущий MVP-корпус:

```text
vega_lite
```

---

## 2. Роли директорий

### `rag_corpus/raw/`

Сырой backup-слой.

```text
rag_corpus/raw/
```

Правило:

```text
raw не используется напрямую в VisRAG runtime.
raw можно пересоздать загрузочными скриптами.
raw может оставаться локальным backup.
```

---

### `rag_corpus/cleaned/`

Ручной curated-слой.

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

Содержит полный normalized record:

```text
id
source
source_path
file_name
file_stem
title
corpus_type
instruction
mark_type
chart_pattern
field_roles
retrieval_text
spec_template
raw_spec
data_policy
removed_data_sections
metadata
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
offline retrieval evaluation / optimization.
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

Формат одной записи:

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

### Parquet

Используется для:

```text
AutoRAG corpus dataset
AutoRAG QA dataset
offline retrieval evaluation
```

---

## 4. Порядок скриптов

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
```

---

## 5. Быстрый запуск pipeline от cleaned

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

Exporter делает:

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

---

## 7. Runtime-safe semantic roles

```text
quantitative
temporal
nominal
ordinal
boolean
geojson
```

---

## 8. Runtime-safe chart types

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

## 9. Почему `data` не хранится в `spec_template`

В ViRAGE данные подключаются runtime-слоем.

ChartGenerator вставляет:

```json
{
  "data": {
    "url": "<prepared_dataset_path>"
  }
}
```

Правило:

```text
spec_template = visual structure only.
data.url attaches later by ChartGeneratorService.
```

---

## 10. Pytest integration tests

### Runtime corpus

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

### Core search

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

---

### Chart pipeline

```text
tests/integration/test_visrag_chart_pipeline_real_corpus.py
```

Проверяет:

```text
VisRAGService
→ ChartGeneratorService
→ SpecValidatorService
```

---

### Render pipeline

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

---

### Intent quality

```text
tests/integration/test_visrag_intent_quality_real_corpus.py
```

Проверяет intent-level correctness:

```text
query
data_profile
selected_fields
expected chart_family
expected field_mapping
expected encoding
```

---

## 11. Unit tests for grounding policy

```text
tests/unit/test_visrag_grounding_policy.py
tests/unit/test_visrag_field_grounding_policy.py
```

Проверяют:

```text
auto policy resolver
prefer behavior
strict selected_fields behavior
soft_fail decision mode
override handling
```

---

## 12. Команды проверки

```powershell
pytest tests -q
```

По отдельности:

```powershell
pytest tests\unit\test_visrag_grounding_policy.py -q
pytest tests\unit\test_visrag_field_grounding_policy.py -q
pytest tests\integration\test_visrag_runtime_corpus_real.py -q
pytest tests\integration\test_visrag_core_search_real_corpus.py -q
pytest tests\integration\test_visrag_chart_pipeline_real_corpus.py -q
pytest tests\integration\test_visrag_render_pipeline_real_corpus.py -q
pytest tests\integration\test_visrag_intent_quality_real_corpus.py -q
```

---

## 13. Текущий зелёный baseline

Подтверждён путь:

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
→ intent-level tests
```

Подтверждённые рабочие кейсы:

```text
scatter_relationship_numeric
line_trend_over_time
histogram_distribution
bar temporal technical smoke
```

---

## 14. Current expected test status

Ожидаемый статус после strict-policy инкремента:

```text
green suite
bar_compare_categories = xfail
heatmap_two_dimensions = xfail
```

`bar_compare_categories` остаётся xfail осознанно.

---

## 15. Known issues

### 15.1. `bar_compare_categories`

Статус:

```text
xfail
```

Причина:

```text
strict selected_fields работает корректно,
но в текущем runtime corpus нет подходящего categorical bar template.
```

Текущее решение:

```text
не добавлять categorical bar template в MVP.
зафиксировать как known issue.
```

---

### 15.2. `heatmap_two_dimensions`

Статус:

```text
xfail
```

Причина:

```text
heatmap/rect runtime support пока не стабилен.
```

---

### 15.3. Over-specific line templates

Некоторые line templates могут быть слишком специфичными:

```text
time_output_utc_scale
```

Возможное будущее решение:

```text
ranking penalty за overly-specific templates.
```

---

### 15.4. Runtime corpus пока небольшой

Strict exporter оставляет только runtime-safe subset.

Текущее состояние:

```text
около 40 included examples из 100 normalized records
```

---

## 16. Что не делаем в текущем MVP

```text
не добавляем categorical bar local template
не чиним heatmap
не расширяем corpus за счёт новых источников
не внедряем full soft_fail fallback
не оптимизируем AutoRAG hybrid pipeline
```

---

## 17. Следующие возможные итерации

### Итерация A. Soft-fail fallback

```text
strict first
prefer fallback
caveat
```

### Итерация B. Heatmap support

```text
rect templates
x nominal
y nominal
color quantitative
```

### Итерация C. Ranking penalty

```text
penalty for overly-specific templates
```

### Итерация D. New corpus integration

Подключать новые источники только после отдельного плана:

```text
chart-llm-hf
nlvcorpus
local_cases
chart-llm
draco
compassql
```

---

## 18. Правила добавления новых корпусов

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

Если источник используется в VegaChat benchmark, он должен иметь:

```text
is_eval_leak_sensitive = true
benchmark_group = ...
source_split = ...
```

---

## 19. Current DoD

Текущий MVP считается стабильным, если:

```text
[done] runtime corpus генерируется
[done] runtime corpus загружается
[done] core search работает
[done] chart pipeline работает
[done] render pipeline работает
[done] intent quality baseline работает
[done] auto/strict grounding policy внедрена
[done] tests green
[done] bar_compare_categories documented xfail
[done] heatmap_two_dimensions documented xfail
```
