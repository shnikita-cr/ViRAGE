# ViRAGE

ViRAGE — мультимодальная агентная система анализа научных данных. Проект обрабатывает одну выбранную модальность входа за один запуск:

- `table` — табличный датасет, по которому строятся аналитические графики;
- `image_folder` — папка изображений, по которой вычисляются технические метрики качества изображений и далее используется табличный контур анализа.

Комбинированный вход запрещён: таблица и папка изображений не должны обрабатываться в одном запуске. Если пользователь передал обе модальности, система должна потребовать выбрать только один тип входа.

Рабочий контур для табличного входа: подготовка данных → профиль данных → анализ запроса → VisRAG → генерация Vega-Lite → проверка спецификации → построение PNG → semantic VLM retry → метрики качества → артефакты и отчёты.

Рабочий контур для папки изображений: обход папки → вычисление метаданных и image-quality metrics → CSV с метриками изображений → стандартный табличный контур ViRAGE.

## 1. Основные возможности

- Оркестратор проектного задания с декомпозицией до 3 аналитических подзадач.
- Одиночный `ViRAGEPipeline` для запуска одного графика без оркестратора.
- Streamlit UI с выбором режима запуска.
- Генерация Vega-Lite спецификаций локальными или облачными LLM.
- Проверка Vega-Lite спецификации через Altair/Vega-Lite runtime.
- Semantic VLM retry loop: визуальная проверка графика и повторная генерация при смысловой ошибке.
- VisRAG: retrieval-рекомендации из runtime-корпуса правил визуализации.
- NLV-бенчмарк без оркестратора.
- Перебор локальных моделей по готовым `local_test`-конфигам.
- Перебор моделей по ролям агентов: `reasoning_model × spec_model × vlm_model`.
- Сбор `SpecScore`, `VisionScore`, `empty_chart_rate`, `visualization_error_rate`, времени выполнения и token usage.
- Сохранение пользовательской обратной связи и VLM-feedback в артефакты/корпус feedback.

## 2. Структура проекта

```text
src/                         основной код ViRAGE
  application/               контракты pipeline, конфигурация, результаты, runtime state
  benchmark/                 модели, runner-ы и метрики benchmark-слоя
  domain/                    доменные модели данных, графиков, feedback, RAG и runtime
  graph/                     LangGraph-граф pipeline и его узлы
  infrastructure/            сборка runtime-зависимостей
  llm/                       фабрика LLM, structured response, runtime profile, prompt budget
  orchestrator/              планирование подзадач, image-folder preprocessing, summary
  services/                  сервисы подготовки данных, RAG, генерации, валидации, VLM, feedback
  visrag_core/               ядро retrieval и хранилищ VisRAG

scripts/                     CLI и benchmark-скрипты
  benchmark/runners/chart/   запуск VegaChat-compatible benchmark
  benchmark/runners/model/   NLV model comparison и role-component sweep
  rag_corpus/                загрузка, экспорт и проверка RAG-корпусов

ui/                          Streamlit-интерфейс и TOML-конфигурации
  config/app/                конфиги для UI-запуска
  config/benchmark/          конфиги для benchmark-запуска

demo_data/                   демонстрационные и тестовые данные
external_datasets/           внешние датасеты, например NLV corpus
rag_corpus/                  runtime/AutoRAG/manual/feedback корпуса
benchmarks/                  внутренние benchmark-кейсы
research/                    исследовательские материалы; не production-код
artifacts/                   результаты запусков, отчёты, графики, model calls
```

## 3. Установка окружения

### 3.1. Python

Рекомендуется Python 3.12.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если нужен локальный PyTorch CPU/GPU-стек, используйте отдельный файл:

```powershell
pip install -r requirements-torch.txt
```

`pip_freeze.txt` не является источником актуальных зависимостей и не должен использоваться для восстановления окружения.

### 3.2. Переменные окружения

Скопируйте пример:

```powershell
Copy-Item .env.example .env
```

Заполните только нужные ключи. Для локального Ollama обычно достаточно `base_url = "http://localhost:11434"` в TOML-конфигурации.

## 4. Ollama в Docker

Если проект работает с Docker-контейнером `ollama`, модели устанавливаются внутри контейнера.

### 4.1. Установка моделей

```powershell
docker exec -it ollama ollama pull gemma3:4b
docker exec -it ollama ollama pull gemma4:e2b-it-qat
docker exec -it ollama ollama pull gemma4:e4b-it-qat
docker exec -it ollama ollama pull qwen2.5-coder:7b
docker exec -it ollama ollama pull qwen3-vl:4b
docker exec -it ollama ollama pull qwen2.5vl:latest
docker exec -it ollama ollama pull nomic-embed-text:latest
docker exec -it ollama ollama pull bge-m3:latest
docker exec -it ollama ollama pull mxbai-embed-large:latest
```

Опционально для расширенного тестирования:

```powershell
docker exec -it ollama ollama pull qwen3-vl:8b
```

### 4.2. Проверка через HTTP API

Проверка доступности Ollama выполняется через HTTP API на `localhost`, а не через `ollama run` на хосте.

```powershell
curl http://localhost:11434/api/tags
```

Проверка текстовой модели:

```powershell
curl http://localhost:11434/api/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"gemma3:4b\",\"prompt\":\"Return only JSON: {\\\"ok\\\": true}\",\"stream\":false}"
```

Проверка embedding-модели:

```powershell
curl http://localhost:11434/api/embeddings ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"nomic-embed-text:latest\",\"prompt\":\"test embedding\"}"
```

## 5. Конфигурации

Конфигурации находятся в `ui/config/app` и `ui/config/benchmark`.

### 5.1. Основные app-конфиги

```text
ui/config/app/project.toml
ui/config/app/project.example.toml
ui/config/app/project-gemma3_4b-local.toml
ui/config/app/project-gemma3_12b-local.toml
ui/config/app/project-gemma4.toml
ui/config/app/project-openai.toml
ui/config/app/project-qwen.toml
```

### 5.2. Локальные тестовые конфиги

App-копии:

```text
ui/config/app/local_test_gemma3_4b_safe.toml
ui/config/app/local_test_gemma3_4b_extended.toml
ui/config/app/local_test_gemma4_e2b_qat.toml
ui/config/app/local_test_gemma4_e4b_qat.toml
ui/config/app/local_test_qwen2_5_coder_7b_safe.toml
```

Benchmark-копии:

```text
ui/config/benchmark/local_test_gemma3_4b_safe.toml
ui/config/benchmark/local_test_gemma3_4b_extended.toml
ui/config/benchmark/local_test_gemma4_e2b_qat.toml
ui/config/benchmark/local_test_gemma4_e4b_qat.toml
ui/config/benchmark/local_test_qwen2_5_coder_7b_safe.toml
```

Для `local_test_*` ожидаемые настройки:

```toml
[settings]
gpu_ram_gb = 8.0
visrag_enabled = true
spec_generation_max_attempts = 20
spec_generation_response_parse_retries = 20
semantic_feedback_loop_enabled = true
semantic_feedback_max_attempts = 20
analytics_tail_enabled = false
enable_evaluation_summary = false
```

Для всех ролей локальных моделей:

```toml
max_output_tokens = 1000
base_url = "http://localhost:11434"
```

## 6. Роли моделей

| Роль | Назначение |
|---|---|
| `reasoning_model` | Планирование, анализ пользовательского запроса, выбор подзадач, structured JSON. |
| `spec_model` | Генерация Vega-Lite спецификаций. |
| `vlm_model` | Визуальная проверка графика, semantic retry, image analysis. |
| `visrag_embedding_model` | Эмбеддинги для VisRAG/AutoRAG. |

### 6.1. All-in-one модели

Одна мультимодальная модель используется во всех ролях:

```text
reasoning_model = X
spec_model = X
vlm_model = X
```

Кандидаты:

```text
gemma3:4b
gemma4:e2b-it-qat
gemma4:e4b-it-qat
qwen3-vl:4b
qwen2.5vl:latest
```

Опционально:

```text
qwen3-vl:8b
```

### 6.2. Role sweep кандидаты

```text
reasoning_candidates:
  gemma3:4b
  gemma4:e2b-it-qat
  gemma4:e4b-it-qat
  qwen2.5-coder:7b
  qwen3-vl:4b

spec_candidates:
  qwen2.5-coder:7b
  gemma4:e2b-it-qat
  gemma4:e4b-it-qat
  gemma3:4b
  qwen3-vl:4b

vlm_candidates:
  gemma3:4b
  gemma4:e2b-it-qat
  gemma4:e4b-it-qat
  qwen3-vl:4b
  qwen2.5vl:latest

embedding_candidates:
  nomic-embed-text:latest
  bge-m3:latest
  mxbai-embed-large:latest
```

## 7. Runtime-профиль моделей

`ModelRuntimeProfile` должен подбирать параметры по модели и `settings.gpu_ram_gb`. Базовое значение:

```toml
gpu_ram_gb = 8.0
```

Для Ollama-вызовов должны передаваться:

```text
num_ctx
num_predict
```

Цель локального режима — не максимальный контекст, а стабильная работа в пределах доступной GPU RAM.

## 8. Prompt budget

Промпты для локальных моделей должны помещаться в безопасный бюджет, особенно для режима 4096 токенов.

Запрещено молчаливое обрезание промпта. При превышении бюджета сжимаются отдельные секции:

1. RAG guidance.
2. Примеры значений.
3. Статистика.
4. История.
5. Validation error.

Нельзя удалять:

- пользовательский запрос;
- список доступных полей;
- схему ожидаемого ответа.

В `model calls` должны сохраняться:

```text
num_ctx
estimated_prompt_tokens
prompt_budget_tokens
was_compressed
compression_notes
```

## 9. Режимы входа

### 9.1. Таблица

Табличный режим принимает путь к таблице. Поддерживаемые форматы зависят от `FrameReader`, базово используются CSV/Excel/Parquet-подобные сценарии через pandas/openpyxl/pyarrow.

Таблица проходит этапы:

```text
read table → data preparation → data profile → query request analysis → VisRAG → Vega-Lite generation → validation → render PNG → semantic feedback loop → report
```

### 9.2. Папка изображений

Визуальный режим принимает папку изображений. `ImageFolderPreprocessor` обходит поддерживаемые форматы:

```text
.bmp .gif .jpeg .jpg .png .tif .tiff .webp
```

Для изображений вычисляются технические признаки:

- имя и путь файла;
- размер файла;
- ширина и высота;
- aspect ratio;
- цветовой режим;
- яркость;
- контраст;
- clipping/saturation ratios;
- edge density;
- laplacian variance;
- noise estimate;
- SNR estimate;
- no-reference IQA scores, если доступен backend.

Результат препроцессинга сохраняется как таблица `image_quality_metrics.csv`, после чего используется стандартный табличный pipeline.

## 10. Streamlit UI

Запуск из корня проекта:

```powershell
streamlit run ui/app.py
```

UI должен поддерживать:

- выбор конфигурации TOML;
- выбор режима запуска: orchestrator или single pipeline;
- настройку `max_charts` для orchestrator-режима;
- настройку количества попыток генерации спецификации;
- настройку semantic VLM attempts;
- отображение шагов выполнения;
- отображение подзадач;
- отображение графика и спецификации;
- таблицы `model calls`;
- сохранение пользовательской обратной связи по графику;
- повторную генерацию конкретного графика с учётом feedback.

## 11. Запуск pipeline из CLI

Одиночный запуск:

```powershell
python scripts\run_pipeline.py --help
```

Оркестратор:

```powershell
python scripts\run_orchestrator.py --help
```

Рекомендуется запускать команды из корня проекта.

## 12. NLV benchmark

NLV-бенчмарк должен запускаться без оркестратора и использовать одиночный `ViRAGEPipeline`.

Путь к NLV corpus:

```text
external_datasets/nlv_corpus
```

Короткий прогон по 20 случайным примерам:

```powershell
python scripts\benchmark\runners\model\run_nlv_model_comparison.py --limit 20 --seed 42 --continue-on-error
```

Полный прогон:

```powershell
python scripts\benchmark\runners\model\run_nlv_model_comparison.py --resume --continue-on-error
```

Прямой VegaChat-compatible runner:

```powershell
python scripts\benchmark\runners\chart\run_vegachat_compatible_benchmark.py --cases external_datasets\nlv_corpus --config ui\config\benchmark\local_test_gemma3_4b_safe.toml --output-dir artifacts\benchmarks\smoke --nlv-mode single_turn --limit 20 --shuffle --seed 42 --disable-analytics-tail
```

### 12.1. Поведение при ошибках

- Ошибка одного кейса должна записываться в результат кейса и не должна останавливать весь прогон.
- Ошибка одной модели не должна останавливать весь цикл при `--continue-on-error`.
- При `--limit` кейсы перемешиваются через `--shuffle --seed`.
- Без `--limit` должны использоваться все кейсы.

### 12.2. Метрики

Ожидаемые агрегаты в `benchmark_report.json`:

```text
total_cases
successful_cases
failed_cases
visualization_error_rate
empty_chart_rate
mean_spec_score
mean_spec_score_failure_as_zero
median_spec_score
mean_vision_score
mean_vision_score_failure_as_zero
median_vision_score
mean_duration_seconds
p95_duration_seconds
mean_prompt_tokens
mean_completion_tokens
mean_total_tokens
total_tokens
chart_text_consistency_rate
stratified_metrics
vegachat_metrics
```

## 13. Role-component sweep

Перебор ролей:

```powershell
python scripts\benchmark\runners\model\run_nlv_role_component_sweep.py --limit 20 --seed 42 --continue-on-error
```

Скрипт создаёт временные TOML-конфиги в:

```text
artifacts/model_nlv_role_sweep/configs/
```

Результаты каждой комбинации сохраняются в отдельной директории:

```text
artifacts/model_nlv_role_sweep/<role-combination>/
```

По умолчанию перебираются:

```text
reasoning_model × spec_model × vlm_model
```

## 14. VisRAG

Runtime-корпус:

```text
rag_corpus/runtime/guidance_chunks.jsonl
```

Chroma index:

```text
resources/chroma/virage_guidance_chunks_nomic_embed_text_latest
```

Сборка runtime Chroma index:

```powershell
python -m scripts.rag_corpus.runtime.build_runtime_chroma_index ^
  --chunks rag_corpus/runtime/guidance_chunks.jsonl ^
  --persist-dir resources/chroma/virage_guidance_chunks_nomic_embed_text_latest ^
  --collection virage_guidance_chunks_nomic_embed_text_latest ^
  --embedding-provider ollama ^
  --embedding-model nomic-embed-text:latest ^
  --embedding-base-url http://localhost:11434 ^
  --batch-size 8 ^
  --max-input-chars 1600 ^
  --recreate
```

AutoRAG-данные находятся отдельно:

```text
rag_corpus/autorag/
```

AutoRAG не является обязательным runtime-шагом.

## 15. Vega-Lite generation и validation

Проект не должен программно исправлять невалидные Vega-Lite спецификации.

Правильный цикл:

```text
model output → parse structured JSON → validate Vega-Lite → if invalid, return validation issue → retry generation with exact issue
```

Запрещено:

- исправлять `mark.type` программно;
- подменять поля;
- нормализовать несовместимые Vega-Lite структуры;
- скрывать ошибку Altair/jsonschema traceback-ом в UI.

Для repeat-графиков требуется единая политика информативных подписей без размытых заголовков:

```text
Repeated metrics
Metric
Value
```

## 16. QueryRequestAnalyzer

`QueryRequestAnalyzer` должен быть устойчив к ответам локальных моделей.

Требования:

- `metric_semantics` не является обязательным;
- неизвестные значения `metric_semantics` нельзя трактовать как конкретную семантику;
- поля таблицы не обязаны быть метриками;
- запрещено привязываться к конкретным именам колонок, агрегатов или примеров;
- `ranking_strategy` может приходить строкой или объектом с ключом `ranking_strategy`;
- `scale_strategy` может приходить строкой или объектом с ключом `scale_strategy`;
- приведение структуры допускается только для полей, уже возвращённых моделью;
- запрещено создавать анализ запроса с нуля;
- пустой JSON без запроса, полей, связей и плана должен отклоняться.

## 17. Semantic VLM retry и analytics tail

`analytics_tail_enabled = false` должен отключать только финальный аналитический хвост:

- финальный VLM-анализ PNG;
- evaluation summary.

Он не должен отключать:

- semantic VLM retry loop;
- visual judge;
- feedback для повторной генерации;
- сохранение rejected specs, если оно включено.

## 18. Пользовательская обратная связь

После построения графика UI должен позволять:

- написать обратную связь по конкретному графику;
- сохранить feedback как артефакт запуска;
- сохранить feedback в корпус обратной связи;
- запустить повторную генерацию конкретного графика с этим feedback.

Feedback от semantic VLM retry также должен сохраняться в feedback-корпус, если это включено настройками.

## 19. Артефакты запуска

Типичные артефакты pipeline:

```text
artifacts/<run_id>/
  chart_plan.json
  run_report.json
  plot.png
  plot_rendering.json
  scenegraph_check.json
  visrag.json
  model_calls.csv
  stage_execution_logs.json
  manual_feedback/
```

Benchmark-артефакты:

```text
artifacts/model_nlv/<config_name>/
  benchmark_results.csv
  benchmark_report.json
  benchmark_report.md
  cases/<case_id>/result.json
  logs/stdout.txt
  logs/stderr.txt
  model_run_status.json
```

Role sweep артефакты:

```text
artifacts/model_nlv_role_sweep/
  role_sweep_runs.json
  configs/*.toml
  <role-combination>/benchmark_report.json
```

## 20. Проверка проекта

Базовые проверки:

```powershell
python -m compileall -q src scripts ui tests
pytest -q tests
```

Проверка CLI:

```powershell
python scripts\benchmark\runners\chart\run_vegachat_compatible_benchmark.py --help
python scripts\benchmark\runners\model\run_nlv_model_comparison.py --help
python scripts\benchmark\runners\model\run_nlv_role_component_sweep.py --help
```

Проверка UI:

```powershell
streamlit run ui/app.py
```

Короткая NLV-проверка:

```powershell
python scripts\benchmark\runners\model\run_nlv_model_comparison.py --limit 20 --seed 42 --continue-on-error
```

Role sweep проверка:

```powershell
python scripts\benchmark\runners\model\run_nlv_role_component_sweep.py --limit 20 --seed 42 --continue-on-error
```

## 21. Проверка результатов NLV

PowerShell-сводка по `benchmark_report.json`:

```powershell
Get-ChildItem artifacts\model_nlv -Recurse -Filter benchmark_report.json |
  ForEach-Object {
    $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
    [PSCustomObject]@{
      config = $_.Directory.Name
      total_cases = $j.total_cases
      failed_cases = $j.failed_cases
      mean_spec_score = $j.mean_spec_score
      mean_spec_score_failure_as_zero = $j.mean_spec_score_failure_as_zero
      mean_vision_score = $j.mean_vision_score
      mean_vision_score_failure_as_zero = $j.mean_vision_score_failure_as_zero
      empty_chart_rate = $j.empty_chart_rate
      visualization_error_rate = $j.visualization_error_rate
    }
  } | Format-Table -AutoSize
```

## 22. Запрещённые практики

В рабочем коде запрещены:

- скрытый fallback;
- программная нормализация невалидной Vega-Lite спецификации;
- молчаливое обрезание промпта;
- `print(...)` вместо `logging`;
- `assert` для проверки входных данных;
- пустой `except`;
- `except Exception` без конкретной обработки;
- `sys.path.insert`;
- самодельные аналоги `_simple_load_dotenv`;
- комбинированный вход `table + image_folder`;
- удаление `demo_data` и `research`.

## 23. Текущий контрольный прогон README-итерации

На текущем архиве были выполнены проверки без изменения Python-кода:

```powershell
python -m compileall -q src scripts ui tests
pytest -q tests
python scripts\benchmark\runners\chart\run_vegachat_compatible_benchmark.py --help
python scripts\benchmark\runners\model\run_nlv_model_comparison.py --help
python scripts\benchmark\runners\model\run_nlv_role_component_sweep.py --help
```

Результат тестов:

```text
167 passed, 2 skipped
```

Проверка `streamlit run ui/app.py` в этой среде не выполнялась, потому что в sandbox-окружении отсутствует пакет `streamlit`. В проектном окружении он должен быть установлен через `requirements.txt`.

## 24. Известные задачи для следующей кодовой итерации

- Сделать `metric_semantics` необязательным в `QueryRequestAnalyzer` и не интерпретировать неизвестные значения как доменную семантику.
- Приводить `ranking_strategy` и `scale_strategy`, если модель вернула объект с одноимённым ключом.
- Убрать жёсткое требование severity-семантики там, где поля таблицы не являются метриками со специальным смыслом.
- Добавить тесты на устойчивость `QueryRequestAnalyzer` к локальным моделям.
- Проверить, что `analytics_tail_enabled = false` не отключает semantic retry.
- Добавить all-in-one local_test-конфиги для `qwen3-vl:4b` и `qwen2.5vl:latest`, если они должны участвовать в all-in-one сравнении.
- Расширить role sweep кандидатами `qwen3-vl:4b` и `qwen2.5vl:latest` для VLM-роли.
- Проверить прямой запуск всех benchmark-скриптов без `PYTHONPATH` и без `sys.path.insert`.
- Разбить крупные функции `semantic_decision_node`, `vegachat_backend.generate`, `visrag_core.engine._generate_response`, `build_pipeline_graph`, `data_profiler.invoke`, `vlm_analysis_node`.
- Убрать дублирование `_is_vega_lite_spec_payload`, `_spec_add_data`, `title_text`, `canonicalize_chart_type`, `normalize_aggregate`, `_mean`, `_mean_bool`, `_percentile`, `_write_json`, `_write_jsonl`, `_read_jsonl`.
