# Тестирование ViRAGE через InfiAgent-DABench / DAEval

Эта инструкция описывает запуск ViRAGE как `chart_grounded` аналитического агента.

Смысл режима:

    вопрос + CSV
    → ViRAGE строит график
    → PNG-only VLM judge проверяет график
    → если график не принят, pipeline делает retry
    → если retry закончились и график не принят, кейс считается failed
    → если график принят, VLMAnalysisService отвечает только по изображению графика
    → ответ сравнивается с эталоном hybrid-оценкой

Финальный анализатор не должен читать таблицу напрямую. Таблица используется только для построения графика.

## 1. Ожидаемая структура датасета

По умолчанию скрипты ожидают InfiAgent в папке:

    Datasets/InfiAgent

Внутри используются конкретные файлы:

    Datasets/InfiAgent/examples/DA-Agent/data/da-dev-questions.jsonl
    Datasets/InfiAgent/examples/DA-Agent/data/da-dev-labels.jsonl
    Datasets/InfiAgent/examples/DA-Agent/data/da-dev-tables/

`da-dev-questions.jsonl` содержит вопросы, имя CSV-файла, ограничения и требуемый формат ответа.

`da-dev-labels.jsonl` содержит эталонные ответы в поле `common_answers`.

`da-dev-tables/` содержит CSV-файлы.

## 2. Проверить структуру датасета

Из корня проекта:

    python scripts/benchmark/infiagent_scan.py

Если датасет лежит не в `Datasets/InfiAgent`:

    python scripts/benchmark/infiagent_scan.py --source-root path/to/InfiAgent

Результат сохраняется сюда:

    artifacts/benchmarks/infiagent_scan.json

## 3. Конвертировать InfiAgent в формат ViRAGE

Команда для основного chart-grounded benchmark:

    python scripts/benchmark/convert_infiagent_dabench.py --chart-answerable-only

Она оставит только задачи, которые можно разумно ответить через статичный график без дополнительных статистических тестов, обучения моделей и точных скрытых вычислений.

Если нужен диагностический полный набор без фильтра:

    python scripts/benchmark/convert_infiagent_dabench.py

Скрипт создаст:

    artifacts/benchmarks/infiagent_cases.jsonl
    artifacts/benchmarks/infiagent_cases.manifest.json

С другим путём к датасету:

    python scripts/benchmark/convert_infiagent_dabench.py --source-root path/to/InfiAgent --output artifacts/benchmarks/infiagent_cases.jsonl

Каждый case получает поля:

    case_id
    dataset_name
    question
    data_path
    expected_answer
    metadata

В `question` добавляются исходный вопрос, constraints и required answer format. Это важно, потому что benchmark-задачи часто требуют точного числового ответа и конкретного формата.

## 4. Smoke-запуск на 3 кейсах

    python scripts/benchmark/run_infiagent_chart_grounded.py --limit 3

Скрипт сам выполнит конвертацию, если не указан `--skip-convert`. По умолчанию он включает chart-answerable фильтр. Для диагностического запуска всех задач используй `--all-cases`.

Результаты будут здесь:

    artifacts/benchmarks/infiagent_chart_grounded/

## 5. Запуск одного кейса

    python scripts/benchmark/run_infiagent_chart_grounded.py --case-id infiagent_0

Повторный запуск только этого кейса:

    python scripts/benchmark/run_infiagent_chart_grounded.py --case-id infiagent_0 --retry-failed

## 6. Полный запуск

    python scripts/benchmark/run_infiagent_chart_grounded.py

Если запуск прервался:

    python scripts/benchmark/run_infiagent_chart_grounded.py --resume

Если нужно перезапустить только упавшие или неуспешные кейсы:

    python scripts/benchmark/run_infiagent_chart_grounded.py --retry-failed

## 7. Готовый generic-запуск без конвертации

Если `artifacts/benchmarks/infiagent_cases.jsonl` уже создан:

    python scripts/benchmark/run_chart_grounded_analysis_benchmark.py --cases artifacts/benchmarks/infiagent_cases.jsonl --config ui/config/project.toml --output-dir artifacts/benchmarks/infiagent_chart_grounded --evaluation-mode hybrid

## 8. Evaluation mode

По умолчанию используется:

    --evaluation-mode hybrid

Режимы:

    none
    rules
    llm
    hybrid

`rules` ищет эталонные токены вида:

    @mean_fare[34.65]

и сравнивает числа с допуском.

`llm` использует reasoning model из конфигурации проекта. Работает и с Ollama, и с OpenAI, потому что использует общий модельный слой ViRAGE.

`hybrid` сначала применяет правила, а если точного совпадения нет — вызывает LLM evaluator.

## 9. Что происходит, если VLM judge не принял график

По умолчанию:

    --failure-policy fail

Это значит:

    график не принят после всех semantic retry → кейс failed

Можно разрешить анализ даже отклонённого графика:

    --failure-policy analyze_anyway

Для основного тестирования лучше оставлять `fail`.

## 10. Где смотреть результаты

Основные файлы:

    artifacts/benchmarks/infiagent_chart_grounded/analysis_benchmark_results.csv
    artifacts/benchmarks/infiagent_chart_grounded/analysis_benchmark_results.json
    artifacts/benchmarks/infiagent_chart_grounded/analysis_benchmark_report.json
    artifacts/benchmarks/infiagent_chart_grounded/analysis_benchmark_failures.md

По каждому кейсу:

    artifacts/benchmarks/infiagent_chart_grounded/cases/<case_id>/result.json

Реальные run-артефакты ViRAGE лежат в `artifact_run_dir`, указанном в `result.json`.

Внутри run-артефактов должны быть:

    input/query.txt
    input/context.json
    nodes/
    model_calls.csv
    stages.csv
    run_status.json

## 11. Инспекция одного кейса

    python scripts/benchmark/inspect_infiagent_case.py --case-id infiagent_0

Скрипт покажет:

    статус
    ошибку
    evaluation score
    expected answer
    путь к изображению
    путь к run-артефактам
    query.txt / context.json / nodes

## 12. Пересборка агрегированного отчёта

Если нужно пересчитать только отчёт по существующим `cases/*/result.json`:

    python scripts/benchmark/evaluate_infiagent_results.py

## 13. Рекомендуемый порядок

Сначала проверить структуру:

    python scripts/benchmark/infiagent_scan.py

Затем конвертировать chart-answerable поднабор:

    python scripts/benchmark/convert_infiagent_dabench.py --chart-answerable-only

Затем smoke:

    python scripts/benchmark/run_infiagent_chart_grounded.py --limit 3

Затем 20 кейсов:

    python scripts/benchmark/run_infiagent_chart_grounded.py --limit 20 --output-dir artifacts/benchmarks/infiagent_chart_grounded_20

Затем retry failed:

    python scripts/benchmark/run_infiagent_chart_grounded.py --limit 20 --output-dir artifacts/benchmarks/infiagent_chart_grounded_20 --retry-failed

После этого полный запуск:

    python scripts/benchmark/run_infiagent_chart_grounded.py --output-dir artifacts/benchmarks/infiagent_chart_grounded_full

## 14. Важные ограничения

InfiAgent-DABench содержит задачи, которые часто требуют точных вычислений по таблице. ViRAGE в этом режиме проверяется как chart-grounded система: он должен строить график и отвечать по графику.

Поэтому основной запуск теперь фильтрует задачи до chart-answerable поднабора. Полный набор можно запускать только как диагностический стресс-тест через `--all-cases`; такие результаты нельзя напрямую трактовать как качество chart-grounded режима.
