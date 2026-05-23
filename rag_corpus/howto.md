# ViRAGE: подготовка RAG-корпуса и benchmark-оценка прироста RAG

Документ описывает практический сценарий для текущей версии ViRAGE:

- подготовить runtime RAG-корпус правил и рекомендаций;
- учесть обновление по ChartSquared / C²;
- измерить прирост от RAG на NLV Corpus;
- отдельно прогнать InfiAgent только на задачах, которые можно решать через построение графика;
- понять, каких частей в репозитории хватает, а какие данные нужно скачать отдельно.

Команды предполагают, что ты находишься в корне проекта:

    D:\programming\projects\ViRAGE

---

# 0. Короткий вывод по текущему репозиторию

В актуальном архиве есть основная инфраструктура для нужного сценария:

- есть benchmark runner для VegaChat/NLV/ChartLLM-совместимых задач:

    scripts/benchmark/run_vegachat_compatible_benchmark.py

- есть отдельные конфиги для сравнения NLV без RAG и с RAG:

    ui/config/project-gemma4-bench_norag.toml
    ui/config/project-gemma4-bench_rag.toml

- эти два конфига отличаются только ключом:

    visrag_enabled = false
    visrag_enabled = true

- есть поддержка NLV Corpus как директории с файлами:

    NLV_Corpus.csv
    vlSpecs.json
    datasets/

- есть InfiAgent chart-grounded benchmark runner:

    scripts/benchmark/run_infiagent_chart_grounded.py

- есть фильтрация InfiAgent до задач, которые можно решать построением графика. По умолчанию `run_infiagent_chart_grounded.py` использует chart-answerable фильтр. Полный набор включается только через `--all-cases`.

- есть pipeline подготовки RAG-корпуса:

    scripts/rag_corpus/run_prepare_corpus.py
    scripts/rag_corpus/run_export_runtime.py
    scripts/rag_corpus/run_export_autorag.py
    scripts/rag_corpus/run_autorag_optimization.py

- есть source extractors для:

    manual_rules
    virage_feedback
    chartsquared
    vega_lite_examples

Чего в архиве нет как готовых данных:

- нет скачанного NLV Corpus;
- нет скачанного InfiAgent-DABench / DAEval;
- нет скачанного C² репозитория;
- нет готового runtime-файла `rag_corpus/runtime/virage_rules.jsonl`.

То есть кодовая часть в основном есть. Внешние датасеты и raw-корпуса нужно скачать отдельно.

---

# 1. Как будем измерять прирост от RAG внутри ViRAGE

RAG в ViRAGE нужно оценивать не как отдельный retrieval benchmark, а через конечное качество построенного графика.

Основная идея:

    один и тот же набор NLV задач
    одна и та же модель
    один и тот же pipeline
    один и тот же порядок кейсов
    отличие только в visrag_enabled

Сравниваем два запуска:

    NLV без RAG  -> ui/config/project-gemma4-bench_norag.toml
    NLV с RAG    -> ui/config/project-gemma4-bench_rag.toml

Главный результат:

    RAG gain = metric(nlv_rag) - metric(nlv_no_rag)

Для метрик ошибок наоборот:

    error reduction = metric(nlv_no_rag) - metric(nlv_rag)

## 1.1. Почему не делаем матрицу по RAG-корпусам

В этой версии RAG-корпуса будут часто меняться: добавляться правила, feedback, C²-материалы, доменные рекомендации. Поэтому не нужно делать большую матрицу вида:

    corpus A x corpus B x retriever x top_k x prompt

Это быстро раздует эксперименты и не даст понятного вывода для диплома.

Правильная схема проще:

1. Зафиксировать baseline без RAG.
2. Зафиксировать текущий runtime RAG-корпус как snapshot.
3. Запустить тот же NLV benchmark с RAG.
4. Сравнить итоговые метрики.
5. Когда корпус изменился, снова экспортировать runtime-корпус и запустить тот же benchmark с новым именем output-dir.

Пример именования:

    artifacts/benchmarks/nlv_no_rag
    artifacts/benchmarks/nlv_rag_20260523_c2_prompts
    artifacts/benchmarks/nlv_rag_20260524_more_feedback

Для отчёта по диплому достаточно сравнения:

    no_rag baseline
    current_rag snapshot

Если нужно показать развитие корпуса, можно добавить одну строку `previous_rag_snapshot`, но не строить большую матрицу.

## 1.2. Основные метрики NLV benchmark

`run_vegachat_compatible_benchmark.py` сохраняет агрегированный отчёт:

    artifacts/benchmarks/<run_name>/benchmark_report.json
    artifacts/benchmarks/<run_name>/benchmark_report.md
    artifacts/benchmarks/<run_name>/benchmark_results.csv

Главные поля из `benchmark_report.json`:

- `visualization_error_rate` — доля кейсов, где спецификация не отрисовалась. Меньше — лучше.
- `empty_chart_rate` — доля пустых/неинформативных графиков. Меньше — лучше.
- `mean_spec_score` — среднее структурное совпадение Vega-Lite spec с эталоном. Больше — лучше.
- `mean_vision_score` — средняя визуальная оценка результата. Больше — лучше.
- `median_spec_score` — медианный spec score. Больше — лучше.
- `median_vision_score` — медианный vision score. Больше — лучше.
- `mean_duration_seconds` — среднее время выполнения кейса. Меньше — лучше, но это вторичная метрика.
- `total_tokens` — суммарный расход токенов. Меньше — дешевле, но это вторичная метрика.

Основные метрики для вывода о приросте RAG:

    mean_spec_score
    mean_vision_score
    visualization_error_rate
    empty_chart_rate

В дипломном тексте можно формулировать так:

    RAG улучшает качество, если при включении visrag_enabled=true растут mean_spec_score/mean_vision_score и одновременно не растут visualization_error_rate/empty_chart_rate.

---

# 2. Быстрая техническая проверка проекта

Перед benchmark нужно проверить, что проект хотя бы компилируется и тесты проходят:

    python -Wdefault -m compileall -q src ui scripts tests

    pytest -q

Для текущего архива проверено:

    53 passed

---

# 3. Установка зависимостей

Базовые зависимости проекта:

    pip install -r requirements.txt

Для тестов:

    pip install -r requirements-dev.txt

Для подготовки RAG-корпуса дополнительно нужны:

    pip install pandas pyarrow requests pydantic pyyaml

Если нужно запускать AutoRAG:

    pip install AutoRAG

---

# 4. Подготовить модель

## 4.1. Ollama

Проверить, что Ollama доступна:

    ollama list

Скачать модель для нормализации корпуса:

    ollama pull qwen2.5-coder:7b

Проверить HTTP API:

    python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json().keys())"

## 4.2. OpenAI

В PowerShell:

    $env:OPENAI_API_KEY="твой_ключ"

Проверить:

    python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"

---

# 5. Скачать внешние источники для RAG-корпуса

## 5.1. Vega-Lite examples

    git clone https://github.com/vega/vega-lite.git rag_corpus\raw\vega_lite_examples\vega-lite

Проверить:

    Test-Path rag_corpus\raw\vega_lite_examples\vega-lite\examples\specs

## 5.2. ChartSquared / C²

C² используется не как источник готовых Vega-Lite spec templates, а как источник идей для feedback/readability/VLM-readability правил.

    git clone https://github.com/chartsquared/C-2.git rag_corpus\raw\chartsquared\C-2

Проверить:

    Test-Path rag_corpus\raw\chartsquared\C-2

## 5.3. NLV Corpus

NLV нужен в первую очередь для benchmark-сравнения `no rag` против `rag`.

    git clone https://github.com/nlvcorpus/nlvcorpus.github.io.git rag_corpus\raw\nlv\nlvcorpus.github.io

Проверить, что структура совместима с loader-ом ViRAGE:

    Test-Path rag_corpus\raw\nlv\nlvcorpus.github.io\NLV_Corpus.csv
    Test-Path rag_corpus\raw\nlv\nlvcorpus.github.io\vlSpecs.json
    Test-Path rag_corpus\raw\nlv\nlvcorpus.github.io\datasets

Если один из путей вернул `False`, нужно найти реальные пути внутри скачанного репозитория и передать в benchmark директорию, где одновременно лежат `NLV_Corpus.csv`, `vlSpecs.json` и `datasets/`.

## 5.4. InfiAgent-DABench / DAEval

По умолчанию скрипты ViRAGE ожидают InfiAgent здесь:

    Datasets\InfiAgent

Внутри должны быть:

    Datasets\InfiAgent\examples\DA-Agent\data\da-dev-questions.jsonl
    Datasets\InfiAgent\examples\DA-Agent\data\da-dev-labels.jsonl
    Datasets\InfiAgent\examples\DA-Agent\data\da-dev-tables\

Проверка:

    python scripts/benchmark/infiagent_scan.py --source-root Datasets\InfiAgent

---

# 6. Обновление по C² в prepare corpus

В текущей версии `run_prepare_corpus.py` поддерживает C² через источник:

    chartsquared

И отдельный параметр глубины обработки:

    --chartsquared-mode prompts_only
    --chartsquared-mode sample
    --chartsquared-mode full

Режимы:

- `prompts_only` — быстрый и безопасный режим. Извлекаются prompt/criteria материалы. Рекомендуется как основной режим для дипломного pipeline.
- `sample` — дополнительно ограниченно просматривает файлы C². Количество ограничивается через `--chartsquared-limit`.
- `full` — полный проход по C². Может быть долгим и шумным, поэтому лучше использовать только для отдельной подготовки корпуса.

Если C² временно не нужен, не надо искать флаг `off`. Достаточно не включать `chartsquared` в `--sources`.

Пример без C²:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources manual_rules virage_feedback vega_lite_examples --clean-processed

Пример с C² в рекомендуемом режиме:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources manual_rules virage_feedback chartsquared vega_lite_examples --chartsquared-mode prompts_only --clean-processed

Пример с C² sample:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources manual_rules virage_feedback chartsquared vega_lite_examples --chartsquared-mode sample --chartsquared-limit 200 --clean-processed

Пример полного C²:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources manual_rules virage_feedback chartsquared vega_lite_examples --chartsquared-mode full --clean-processed

---

# 7. Очистить старые результаты подготовки корпуса

Не удаляй `rag_corpus\raw\manual_rules`, если там уже есть новые seed rules.

Можно удалить generated-результаты:

    Remove-Item -Recurse -Force rag_corpus\extracted -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\processed\llm_normalized -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.deduped.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.validated.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\normalization_failures.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\autorag\virage_rules -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\runtime -ErrorAction SilentlyContinue

Заново создать папки:

    New-Item -ItemType Directory -Force rag_corpus\extracted
    New-Item -ItemType Directory -Force rag_corpus\processed\llm_normalized
    New-Item -ItemType Directory -Force rag_corpus\autorag\virage_rules
    New-Item -ItemType Directory -Force rag_corpus\runtime

---

# 8. Проверить raw-источники

    python scripts/rag_corpus/sources/scan_sources.py

После этого должны появиться:

    rag_corpus/manifests/raw_inventory.json
    rag_corpus/manifests/source_inventory.md

Открыть отчёт:

    notepad rag_corpus\manifests\source_inventory.md

---

# 9. Подготовить RAG-корпус

## 9.1. Smoke-подготовка на 10 записях

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --limit 10 --sources manual_rules virage_feedback chartsquared vega_lite_examples --chartsquared-mode prompts_only --clean-processed

## 9.2. Полная подготовка через Ollama

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources manual_rules virage_feedback chartsquared vega_lite_examples --chartsquared-mode prompts_only --clean-processed

Если Ollama не на стандартном порту:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --base-url http://localhost:11434 --sources manual_rules virage_feedback chartsquared vega_lite_examples --chartsquared-mode prompts_only --clean-processed

## 9.3. Продолжить после обрыва

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --resume

## 9.4. Повторить только упавшие записи

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --retry-failed

## 9.5. Подготовка через OpenAI

    python scripts/rag_corpus/run_prepare_corpus.py --provider openai --model gpt-4.1-mini --sources manual_rules virage_feedback chartsquared vega_lite_examples --chartsquared-mode prompts_only --clean-processed

Ожидаемые файлы:

    rag_corpus/extracted/manual_rules.jsonl
    rag_corpus/extracted/virage_feedback.jsonl
    rag_corpus/extracted/chartsquared.jsonl
    rag_corpus/extracted/vega_lite_examples.jsonl
    rag_corpus/processed/llm_normalized/*.jsonl
    rag_corpus/processed/all_rules.jsonl
    rag_corpus/processed/all_rules.deduped.jsonl
    rag_corpus/processed/all_rules.validated.jsonl
    rag_corpus/processed/processing_report.json
    rag_corpus/processed/processing_report.md

---

# 10. Проверить processed-корпус

Открыть отчёт:

    notepad rag_corpus\processed\processing_report.md

Посмотреть первые записи:

    python -c "import json,itertools; f=open('rag_corpus/processed/all_rules.validated.jsonl',encoding='utf-8'); [print(json.dumps(json.loads(x),ensure_ascii=False,indent=2)[:1200]) for x in itertools.islice(f,3)]"

Проверить, что в processed-записях нет Vega-Lite spec-полей:

    python -c "from pathlib import Path; text=Path('rag_corpus/processed/all_rules.validated.jsonl').read_text(encoding='utf-8'); print(any(x in text for x in ['\"mark\"','\"encoding\"','\"$schema\"','\"spec_template\"']))"

Ожидаемый вывод:

    False

---

# 11. Экспортировать runtime-корпус для ViRAGE

    python scripts/rag_corpus/run_export_runtime.py

Ожидаемый файл:

    rag_corpus/runtime/virage_rules.jsonl

Проверить:

    python -c "from pathlib import Path; p=Path('rag_corpus/runtime/virage_rules.jsonl'); print(p.exists(), p.stat().st_size if p.exists() else 0)"

Если файл есть и размер больше нуля, `VisRAGService` сможет читать runtime rules через JSONL backend.

---

# 12. Опционально: AutoRAG export и optimization

AutoRAG здесь нужен только для настройки retrieval-компонентов. Для основного измерения прироста RAG в ViRAGE он не обязателен.

Экспортировать корпус для AutoRAG:

    python scripts/rag_corpus/run_export_autorag.py

Ожидаемые файлы:

    rag_corpus/autorag/virage_rules/corpus.parquet
    rag_corpus/autorag/virage_rules/qa.parquet
    rag_corpus/autorag/virage_rules/configs/virage_rules_all.yaml

Проверить parquet:

    python -c "import pandas as pd; print(pd.read_parquet('rag_corpus/autorag/virage_rules/corpus.parquet').head()); print(pd.read_parquet('rag_corpus/autorag/virage_rules/qa.parquet').head())"

Dry-run AutoRAG:

    python scripts/rag_corpus/run_autorag_optimization.py --dry-run

Запуск AutoRAG:

    python scripts/rag_corpus/run_autorag_optimization.py

Собрать результаты:

    python scripts/rag_corpus/autorag/collect_results.py

---

# 13. Benchmark 1: NLV без RAG

## 13.1. Smoke-запуск

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases rag_corpus\raw\nlv\nlvcorpus.github.io --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag_smoke --limit 20

## 13.2. Полный запуск

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases rag_corpus\raw\nlv\nlvcorpus.github.io --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag

## 13.3. Продолжить прерванный запуск

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases rag_corpus\raw\nlv\nlvcorpus.github.io --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag --resume

## 13.4. Перезапустить только failed/missing кейсы

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases rag_corpus\raw\nlv\nlvcorpus.github.io --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag --retry-failed

---

# 14. Benchmark 2: NLV с RAG

Перед запуском проверь, что runtime-корпус существует:

    Test-Path rag_corpus\runtime\virage_rules.jsonl

Если вернул `False`, сначала выполни:

    python scripts/rag_corpus/run_export_runtime.py

## 14.1. Smoke-запуск

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases rag_corpus\raw\nlv\nlvcorpus.github.io --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag_smoke --limit 20

## 14.2. Полный запуск

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases rag_corpus\raw\nlv\nlvcorpus.github.io --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag

## 14.3. Продолжить прерванный запуск

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases rag_corpus\raw\nlv\nlvcorpus.github.io --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag --resume

## 14.4. Перезапустить только failed/missing кейсы

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases rag_corpus\raw\nlv\nlvcorpus.github.io --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag --retry-failed

---

# 15. Сравнить NLV no RAG и NLV RAG

Основные отчёты:

    artifacts\benchmarks\nlv_no_rag\benchmark_report.json
    artifacts\benchmarks\nlv_rag\benchmark_report.json

PowerShell-команда для сравнения:

    @'
    import json
    from pathlib import Path

    no_rag_path = Path('artifacts/benchmarks/nlv_no_rag/benchmark_report.json')
    rag_path = Path('artifacts/benchmarks/nlv_rag/benchmark_report.json')

    no_rag = json.loads(no_rag_path.read_text(encoding='utf-8'))
    rag = json.loads(rag_path.read_text(encoding='utf-8'))

    higher_is_better = [
        'mean_spec_score',
        'mean_vision_score',
        'median_spec_score',
        'median_vision_score',
    ]
    lower_is_better = [
        'visualization_error_rate',
        'empty_chart_rate',
        'mean_duration_seconds',
        'total_tokens',
    ]

    print('metric,no_rag,rag,delta,interpretation')
    for metric in higher_is_better:
        a = no_rag.get(metric)
        b = rag.get(metric)
        delta = None if a is None or b is None else round(b - a, 6)
        print(f'{metric},{a},{b},{delta},higher_is_better')

    for metric in lower_is_better:
        a = no_rag.get(metric)
        b = rag.get(metric)
        delta = None if a is None or b is None else round(a - b, 6)
        print(f'{metric},{a},{b},{delta},lower_is_better_reduction')
    '@ | python -

Интерпретация:

- для `mean_spec_score`, `mean_vision_score`, `median_spec_score`, `median_vision_score` положительный `delta` означает улучшение с RAG;
- для `visualization_error_rate`, `empty_chart_rate`, `mean_duration_seconds`, `total_tokens` положительный `delta` означает снижение ошибки/времени/расхода с RAG;
- `total_tokens` не является качественной метрикой, но нужен для оценки цены улучшения.

---

# 16. Benchmark 3: InfiAgent только chart-answerable задачи

Этот benchmark проверяет ViRAGE как chart-grounded аналитического агента:

    вопрос + CSV
    → ViRAGE строит график
    → VLM judge принимает или отклоняет график
    → финальный анализатор отвечает только по изображению графика
    → ответ сравнивается с эталоном

Важно: InfiAgent содержит много задач, которые требуют точных вычислений по таблице. Для текущей постановки нужны только задачи, которые можно разумно решить построением статичного графика.

В текущем скрипте это поведение включено по умолчанию. Не используй `--all-cases` для основного benchmark.

## 16.1. Проверить датасет

    python scripts/benchmark/infiagent_scan.py --source-root Datasets\InfiAgent

## 16.2. Smoke-запуск на 3 кейсах

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag_smoke --limit 3

## 16.3. Запуск на 20 кейсах

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag_20 --limit 20

## 16.4. Полный chart-answerable запуск

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag

## 16.5. Продолжить прерванный запуск

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag --resume

## 16.6. Перезапустить только failed/missing кейсы

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag --retry-failed

## 16.7. Инспекция одного кейса

    python scripts/benchmark/inspect_infiagent_case.py --output-dir artifacts\benchmarks\infiagent_chartable_rag --case-id infiagent_0

## 16.8. Пересобрать агрегированный отчёт

    python scripts/benchmark/evaluate_infiagent_results.py --output-dir artifacts\benchmarks\infiagent_chartable_rag

Основной отчёт:

    artifacts\benchmarks\infiagent_chartable_rag\analysis_benchmark_report.json

Главные поля:

- `accepted_chart_rate` — доля кейсов, где итоговый график принят judge-ом;
- `mean_evaluation_score` — средняя оценка ответа по эталону;
- `correct_rate` — доля полностью правильных ответов;
- `partial_or_correct_rate` — доля частично или полностью правильных ответов;
- `mean_chart_groundedness` — насколько ответ опирается на график;
- `mean_hallucination_risk` — риск галлюцинаций, меньше — лучше;
- `total_tokens` — расход токенов.

---

# 17. Рекомендуемая последовательность для дипломного эксперимента

## 17.1. Проверка проекта

    python -Wdefault -m compileall -q src ui scripts tests
    pytest -q

## 17.2. Подготовка RAG-корпуса

    python scripts/rag_corpus/sources/scan_sources.py

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources manual_rules virage_feedback chartsquared vega_lite_examples --chartsquared-mode prompts_only --clean-processed

    python scripts/rag_corpus/run_export_runtime.py

## 17.3. NLV no RAG

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases rag_corpus\raw\nlv\nlvcorpus.github.io --config ui\config\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag

## 17.4. NLV RAG

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases rag_corpus\raw\nlv\nlvcorpus.github.io --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\nlv_rag

## 17.5. InfiAgent chartable

    python scripts/benchmark/infiagent_scan.py --source-root Datasets\InfiAgent

    python scripts/benchmark/run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\project-gemma4-bench_rag.toml --output-dir artifacts\benchmarks\infiagent_chartable_rag

## 17.6. Сравнение NLV

    @'
    import json
    from pathlib import Path

    pairs = {
        'no_rag': json.loads(Path('artifacts/benchmarks/nlv_no_rag/benchmark_report.json').read_text(encoding='utf-8')),
        'rag': json.loads(Path('artifacts/benchmarks/nlv_rag/benchmark_report.json').read_text(encoding='utf-8')),
    }

    metrics = [
        'total_cases',
        'successful_cases',
        'failed_cases',
        'visualization_error_rate',
        'empty_chart_rate',
        'mean_spec_score',
        'mean_vision_score',
        'median_spec_score',
        'median_vision_score',
        'mean_duration_seconds',
        'total_tokens',
    ]

    print('metric,no_rag,rag,raw_delta_rag_minus_no_rag')
    for metric in metrics:
        a = pairs['no_rag'].get(metric)
        b = pairs['rag'].get(metric)
        delta = None if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) else round(b - a, 6)
        print(f'{metric},{a},{b},{delta}')
    '@ | python -

---

# 18. Что считать готовностью результата

Минимальная готовность RAG-корпуса:

- `rag_corpus/processed/all_rules.validated.jsonl` существует;
- `rag_corpus/runtime/virage_rules.jsonl` существует;
- в runtime JSONL нет `mark`, `encoding`, `$schema`, `spec_template` как spec-template полей;
- есть записи нескольких типов:

    chart_pattern
    readability_rule
    scale_plot_area_rule
    vlm_readability_rule
    domain_semantics_rule

Минимальная готовность NLV-сравнения:

- `artifacts/benchmarks/nlv_no_rag/benchmark_report.json` существует;
- `artifacts/benchmarks/nlv_rag/benchmark_report.json` существует;
- оба запуска выполнены на одном и том же наборе кейсов;
- `total_cases` совпадает;
- сравнение показывает delta по главным метрикам.

Минимальная готовность InfiAgent chartable benchmark:

- `python scripts/benchmark/infiagent_scan.py --source-root Datasets\InfiAgent` не показывает критических ошибок;
- `artifacts/benchmarks/infiagent_chartable_rag/analysis_benchmark_report.json` существует;
- запуск сделан без `--all-cases`;
- в отчёте есть `accepted_chart_rate`, `mean_evaluation_score`, `correct_rate`, `partial_or_correct_rate`.

---

# 19. Что ещё не хватает для полностью удобной оценки

В текущем репозитории нет отдельного скрипта `compare_benchmark_reports.py`, который красиво сравнивает два `benchmark_report.json` и сохраняет CSV/MD отчёт. Сейчас сравнение можно делать PowerShell-командой из раздела 15.

Если нужно улучшить репозиторий следующей итерацией, стоит добавить:

    scripts/benchmark/compare_vegachat_reports.py

Он должен принимать:

    --baseline artifacts/benchmarks/nlv_no_rag/benchmark_report.json
    --candidate artifacts/benchmarks/nlv_rag/benchmark_report.json
    --output artifacts/benchmarks/nlv_rag_gain.md

Но для текущей задачи это не блокер: запуск benchmark и сравнение уже возможны.
