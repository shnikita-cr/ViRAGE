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
    python scripts/rag_corpus/export_guidance_chunks.py

AutoRAG используется offline для выбора retrieval-конфигурации. Runtime использует уже выбранный config и компактный `guidance_chunks.jsonl`.


## Оркестратор аналитических задач

Оркестратор добавляет верхний слой над single-chart pipeline. Он не выбирает финальный тип графика и не заменяет RAG. Его задача — раскрыть общий пользовательский запрос в 1–3 аналитические подзадачи, которые затем могут быть выполнены обычным ViRAGE pipeline.

Логика разделения ответственности:

    AnalysisPlanner решает, что анализировать.
    RAG Engine подбирает guidance для выбранной подзадачи.
    Spec Generator строит Vega-Lite спецификацию.

Plan-only запуск строит план через reasoning LLM и не выполняет subtask pipeline:

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
- plan-only режим нужен для быстрой проверки `chart_plan.json` без запуска subtask pipeline; сам план строится через reasoning LLM.

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

`semantic` и `hybrid` требуют полного `guidance_chunk_embeddings.jsonl`; при отсутствии embeddings запуск завершается ошибкой подготовки RAG. `lexical_bm25` используется только как явно заданный backend/baseline и не включается автоматически.

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

## Step 7: Image-folder mode

The orchestrator can now treat an image directory as a lightweight multimodal input. It does not perform domain diagnosis from image content. Instead, it converts the folder into a tabular no-reference image-quality dataset and then reuses the existing ViRAGE tabular pipeline.

Input flow:

    image folder
    → input/image_folder/image_quality_metrics.csv
    → DataProfiler
    → AnalysisPlanner
    → up to 3 ViRAGE subtasks

Generated image-folder artifacts:

    artifacts/<run_id>/input/image_folder/image_quality_metrics.csv
    artifacts/<run_id>/input/image_folder/failed_images.csv
    artifacts/<run_id>/input/image_folder/image_preprocessing_report.json

Main extracted fields:

    file_name, file_path, relative_path, group, extension, width, height, aspect_ratio, file_size_bytes, channels, mean_brightness, std_brightness, dynamic_range, contrast_rms, michelson_contrast, dark_pixel_ratio, bright_pixel_ratio, underexposure_ratio, overexposure_ratio, shadow_clipping_ratio, highlight_clipping_ratio, clipping_ratio, saturation_ratio, exposure_balance_score, entropy, edge_density, laplacian_variance, tenengrad_score, noise_estimate, snr_estimate, brisque_score, niqe_score, piqe_score

Supported image extensions:

    .png, .jpg, .jpeg, .tif, .tiff, .bmp, .gif, .webp

Exposure metrics:

    underexposure_ratio = count(I < 30) / N
    overexposure_ratio = count(I > 225) / N
    clipping_ratio = count(I <= 1 or I >= 254) / N
    exposure_balance_score = 1 - abs(mean_brightness - 127.5) / 127.5

No-reference IQA metrics:

    brisque_score = pyiqa.create_metric("brisque")
    niqe_score = pyiqa.create_metric("niqe")
    piqe_score = pyiqa.create_metric("piqe")

BRISQUE, NIQE, and PIQE are computed through the external `pyiqa` library. Lower values mean better perceived image quality. The image-folder mode does not compute PSNR, SSIM, MSE, LPIPS, or other reference-based metrics because they require a ground-truth/reference image.

Run plan-only image-folder analysis:

    python scripts/run_orchestrator.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --query "Проанализируй качество изображений и найди проблемные файлы" --data-path path/to/images --input-type image_folder --run-id orch_images_plan

Run with subtask execution:

    python scripts/run_orchestrator.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --query "Проанализируй качество изображений и найди проблемные файлы" --data-path path/to/images --input-type image_folder --run-id orch_images_execute --execute

`--input-type auto` treats directories as image folders and files as ordinary tables.

## Publication-quality VLM criteria

Visual judge now includes additional publication-oriented criteria. These scores are not separate primary pipeline metrics; they are additional VLM judge criteria saved in `visual_chart_judge.json` and propagated to `run_report.json`:

    plot_area_usage_score
    axis_domain_score
    layout_compactness_score
    repeat_axis_label_score
    publication_layout_score

The criteria target known failure patterns:

- data variation occupies only a small part of the visible plot area;
- an axis starts at zero or uses a very wide domain when zero is not visually justified by the task;
- a small number of categories is spread across an unnecessarily wide canvas;
- repeat/facet charts have ambiguous axis labels or missing metric names in panel headers;
- the static chart would require manual cropping, relabeling, or layout repair before use in a paper.

The judge also returns issue lists:

    plot_area_issues
    axis_domain_issues
    layout_compactness_issues
    repeat_axis_label_issues
    publication_layout_issues

These values are VLM-side criteria. A deterministic spec/data check for axis domains can be added later, but the current step intentionally keeps these checks inside the visual judge.

## Manual publication-quality seed rules

The project includes non-duplicative seed rules for recurrent publication-quality chart problems:

    rag_corpus/manual_sources/scientific_figure_guidance/virage_publication_quality_rules.txt

They cover four specific issue types instead of repeating generic corpus rules:

    axis_domain_issue
    compact_categorical_layout_issue
    repeat_axis_label_issue
    publication_layout_issue

When `load_scientific_figure_guidance.py --refresh` is run, manual seed files from `rag_corpus/manual_sources/scientific_figure_guidance` are copied into the generated `scientific_figure_guidance` raw corpus. Browser-saved manual caches for protected external pages remain separate from these seed rules.

## Manual vulnerability tests

Use the prompts in:

    docs/virage_manual_vulnerability_queries.md

They are designed to expose remaining weaknesses in axis domains, plot area usage, compact categorical layout, repeat/facet labels, image-folder metrics, and orchestrator task planning.

## Feedback corpus preparation

ViRAGE stores raw user/VLM chart feedback as an append-only log. This raw log is not used directly by runtime RAG. Feedback must be normalized, reviewed, and exported as approved `manual_feedback` / `vlm_feedback` chunks.

Main command:

    python scripts/rag_corpus/export_feedback_chunks.py --mode llm --config ui/config/benchmark/project-gemma4-bench_rag.toml --raw rag_corpus/feedback/visual_feedback.jsonl --normalized rag_corpus/feedback/normalized_feedback.jsonl --output rag_corpus/feedback/manual_feedback_chunks.jsonl

Explicit rules mode is available for offline bootstrapping and unit tests:

    python scripts/rag_corpus/export_feedback_chunks.py --mode rules --raw rag_corpus/feedback/visual_feedback.jsonl --normalized rag_corpus/feedback/normalized_feedback.jsonl --output rag_corpus/feedback/manual_feedback_chunks.jsonl

Approved feedback can be combined with the current guidance corpus into a temporary AutoRAG variant:

    python scripts/rag_corpus/export_feedback_chunks.py --mode rules --normalized-only --approve-all --normalized rag_corpus/feedback/normalized_feedback.jsonl --output rag_corpus/feedback/manual_feedback_chunks.jsonl --base-corpus rag_corpus/runtime/guidance_chunks.jsonl --merged-output rag_corpus/feedback/guidance_with_feedback.jsonl

The script never silently adds raw feedback to the runtime RAG corpus.

## Benchmark suites and EDA guidance

ViRAGE benchmark cases are classified by suite, input modality, analytical task, chart family, output target, and known risks. Cases are stored in:

    benchmarks/cases/*.jsonl

Run all E2E cases in plan-only mode:

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --run-id e2e_plan_001

Run all E2E cases with execution:

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --run-id e2e_execute_001 --execute

Run one suite:

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --run-id e2e_eda_001 --suite eda_manual --execute

The runner writes reports to:

    artifacts/<run_id>/e2e_cases_report/

EDA guidance is stored as a separate RAG source kind:

    eda_guidance

Prepare it with:

    python scripts/rag_corpus/loading/load_eda_guidance.py --refresh

Then rebuild runtime chunks and embeddings:

    python scripts/rag_corpus/export_guidance_chunks.py --max-chars 1000 --overlap-chars 120

    python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model nomic-embed-text:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600

See also:

    docs/virage_benchmark_taxonomy.md
    docs/virage_eda_guidance.md

