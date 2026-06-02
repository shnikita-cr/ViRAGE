# ViRAGE

ViRAGE — система построения Vega-Lite графиков и анализа данных по готовому изображению графика.

Основная идея проекта: пользователь задаёт вопрос к таблице, система строит график, технически проверяет спецификацию, отрисовывает изображение, визуально проверяет соответствие запроса графику и затем выполняет анализ именно по принятому графику.

## Основной конвейер

    data profiling
    query request analysis
    data preparation
    VisRAG retrieval
    Vega-Lite spec generation
    deterministic spec repair
    spec validation
    plot rendering
    scenegraph check
    empty chart check
    PNG-only visual judge
    feedback retry loop
    chart-grounded VLM analysis
    benchmark / evaluation summary

## Генерация спецификаций

По умолчанию используется внутренний backend:

    spec_generation_backend = "vegachat_codegen"

Доступные backend-и:

- `vegachat_codegen` — основной LLM backend генерации Vega-Lite спецификаций.
- `template` — простой backend для тестов и проверки RAG-кандидатов.

Внешние генераторы графиков не подключаются как зависимости проекта. Идеи VegaChat используются внутри собственного генератора ViRAGE: строгий контракт вывода, компактный контекст, учёт ошибок валидации, повторная генерация и детерминированное исправление частых ошибок спецификаций.

## Совместимость моделей

Проект поддерживает локальный запуск через Ollama и запуск через OpenAI.

Пример локальной конфигурации:

    [settings]
    spec_generation_backend = "vegachat_codegen"
    semantic_feedback_loop_enabled = true
    visual_judge_use_chartsquared = true

    [reasoning_model]
    provider = "ollama"
    model = "qwen2.5:7b"
    base_url = "http://localhost:11434"

    [spec_model]
    provider = "ollama"
    model = "qwen2.5-coder:7b"
    base_url = "http://localhost:11434"

    [vlm_model]
    provider = "ollama"
    model = "llava:latest"
    base_url = "http://localhost:11434"

    [vision_judge_model]
    provider = "ollama"
    model = "llava:latest"
    base_url = "http://localhost:11434"

Пример OpenAI-конфигурации:

    [settings]
    spec_generation_backend = "vegachat_codegen"
    semantic_feedback_loop_enabled = true

    [reasoning_model]
    provider = "openai"
    model = "gpt-4.1-mini"

    [spec_model]
    provider = "openai"
    model = "gpt-4.1-mini"

    [vlm_model]
    provider = "openai"
    model = "gpt-4.1-mini"

    [vision_judge_model]
    provider = "openai"
    model = "gpt-4.1-mini"

## Visual judge

Visual judge работает в PNG-only режиме. Он получает только:

- изображение графика;
- исходный запрос пользователя;
- компактные визуальные требования из анализа запроса.

Он не получает Vega-Lite спецификацию, таблицу, профиль данных и технические факты. Если требуемый элемент не виден на изображении, график считается некорректным для пользовательской цели, даже если спецификация технически валидна.

Tooltip не считается достаточным доказательством для статичного изображения.

## ChartSquared / C-2

Проект может использовать локальный архив C-2 как источник идей ChartAF для визуальной проверки: критерии под запрос, вопросы да/нет, визуальная оценка по изображению и конкретная обратная связь для следующей генерации.

ViRAGE не использует оригинальный LLM-wrapper C-2. Все вызовы моделей проходят через собственный слой ViRAGE, поэтому сохраняется совместимость с Ollama и OpenAI.

Путь к локальному проекту можно указать так:

    chartsquared_project_root = "../C-2-main"

## RAG-корпуса

RAG-корпуса готовятся offline и больше не строятся вокруг готовых Vega-Lite spec-шаблонов. VisRAG должен возвращать правила и guidance, а конкретную Vega-Lite спецификацию строит `ChartGeneratorService`.

Основные типы документов:

- `chart_pattern` — какой визуальный паттерн подходит задаче;
- `readability_rule` — как сделать график читаемым;
- `scale_plot_area_rule` — как использовать площадь графика и не дать выбросам сжать основную структуру данных;
- `vlm_readability_rule` — что должно быть видно в статичном PNG для VLM judge и VLM analysis.

Структура корпуса:

    rag_corpus/raw          исходные датасеты и feedback
    rag_corpus/extracted    унифицированные source records
    rag_corpus/processed    LLM-нормализованные rule records
    rag_corpus/autorag      parquet-файлы для AutoRAG
    rag_corpus/runtime      компактный runtime export

Подготовка выполняется через LLM-normalization с Ollama или OpenAI:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b
    python scripts/rag_corpus/run_export_autorag.py
    python scripts/rag_corpus/run_export_runtime.py

AutoRAG используется offline для выбора retrieval-конфигурации. Runtime использует уже выбранный config и компактный `virage_rules.jsonl`.


## Оркестратор аналитических задач

Оркестратор добавляет верхний слой над single-chart pipeline. Он не выбирает финальный тип графика и не заменяет RAG. Его задача — раскрыть общий пользовательский запрос в 1–3 аналитические подзадачи, которые затем могут быть выполнены обычным ViRAGE pipeline.

Логика разделения ответственности:

    AnalysisPlanner решает, что анализировать.
    RAG Engine подбирает guidance для выбранной подзадачи.
    Spec Generator строит Vega-Lite спецификацию.

Plan-only запуск без вызова LLM-моделей:

    python scripts/run_orchestrator.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --query "Проанализируй данные и покажи основные закономерности" --data-path demo_data/Iris.csv --run-id orch_iris_plan

Полный запуск подзадач через pipeline:

    python scripts/run_orchestrator.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --query "Проанализируй данные и покажи основные закономерности" --data-path demo_data/Iris.csv --run-id orch_iris_execute --execute

Основные артефакты:

    artifacts/<run_id>/orchestrator_request.json
    artifacts/<run_id>/data_profile.json
    artifacts/<run_id>/chart_plan.json
    artifacts/<run_id>/orchestrator_report.json
    artifacts/<run_id>/final_summary.md
    artifacts/<run_id>/subruns/<subtask>/run_report.json

Ограничения текущего шага:

- максимум 3 подзадачи;
- планировщик использует только существующие поля таблицы;
- тип графика не фиксируется на этапе планирования;
- plan-only режим нужен для быстрой проверки `chart_plan.json` без затрат на LLM.

## Task-specific RAG guidance

В execute-режиме оркестратор передаёт выбранную аналитическую подзадачу в `PipelineRequest.user_context.analysis_subtask`. Runtime RAG использует этот контракт для retrieval-запроса и prompt guidance:

    AnalysisSubtask
    + QueryRequestAnalysisResult
    + DataProfile
    → task-specific VisRAG guidance

Разделение ответственности:

- `AnalysisPlanner` выбирает, что анализировать;
- `RAG Engine` ищет методические рекомендации для уже выбранной задачи;
- `Spec Generator` строит Vega-Lite спецификацию;
- RAG не заменяет выбранную аналитическую задачу другой задачей.

Runtime lexical fallback отключён. Если `guidance_chunk_embeddings.jsonl` отсутствует или не покрывает все chunks, запуск должен завершиться ошибкой подготовки RAG, а не молча перейти на лексический поиск. BM25 остаётся только как явно заданный AutoRAG baseline.

## Benchmark

Проект тестируется по двум независимым направлениям.

Первое направление — качество построения графиков:

    python scripts/benchmark/run_vegachat_compatible_benchmark.py

Оцениваются:

- valid spec rate;
- visualization error rate;
- empty chart rate;
- Spec Score;
- Encoding F1;
- Full F1;
- Vision Score;
- retries;
- duration;
- token usage.

Второе направление — качество аналитической системы через chart-grounded режим. Для InfiAgent-DABench используется локальная структура:

    Datasets/InfiAgent/examples/DA-Agent/data/da-dev-questions.jsonl
    Datasets/InfiAgent/examples/DA-Agent/data/da-dev-labels.jsonl
    Datasets/InfiAgent/examples/DA-Agent/data/da-dev-tables/

Проверка структуры:

    python scripts/benchmark/infiagent_scan.py

Конвертация только задач, которые можно разумно ответить по графику:

    python scripts/benchmark/convert_infiagent_dabench.py --chart-answerable-only

Запуск benchmark:

    python scripts/benchmark/run_infiagent_chart_grounded.py --limit 10

Финальный анализ должен опираться на принятое изображение графика, а не на прямой расчёт по таблице. Для диагностического полного набора можно добавить `--all-cases`.

## Установка

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    pip install -r requirements-dev.txt

## Запуск UI

    streamlit run ui/app.py

## Проверка

    python -m compileall -q src scripts
    pytest -q

## Структура

    src/application        загрузка конфигурации и окружения
    src/domain             модели данных и контрактов
    src/graph              узлы и маршрутизация конвейера
    src/llm                общий слой вызова моделей
    src/services           основные сервисы ViRAGE
    src/services/spec_generation  генерация Vega-Lite спецификаций
    src/services/visual_feedback  visual judge и feedback loop
    scripts/benchmark      запуск benchmark-сценариев
    scripts/rag_corpus     подготовка RAG-корпусов
    ui                     Streamlit-интерфейс
