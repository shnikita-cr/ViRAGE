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

Серьёзные RAG-корпуса должны строиться из исходных датасетов, а не из вручную подготовленных демонстрационных записей.

Рекомендуемые источники:

- Vega-Lite examples;
- NLV corpus;
- ChartLLM / VL2NL;
- nvBench;
- ChartUIE-8K;
- Draco;
- CompassQL / Voyager.

LLM-processing scripts должны преобразовывать исходники в записи следующих типов:

- chart pattern;
- visual requirement;
- design constraint;
- repair rule;
- judge rule;
- analysis rule.

AutoRAG используется offline для выбора retrieval-конфигурации. Runtime использует уже выбранный конфиг VisRAG.

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
