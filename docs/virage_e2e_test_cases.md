# ViRAGE E2E test cases

Этот набор нужен для ручной и полуавтоматической проверки слабых мест ViRAGE после включения orchestrator, task-specific RAG, scientific guidance, image-folder mode, VLM publication criteria и runtime hybrid RAG.

Файл кейсов:

    benchmarks/virage_e2e_test_cases.jsonl

Демо-таблицы:

    demo_data/test_cases/axis_domain_narrow_range.csv
    demo_data/test_cases/compact_boxplot_small_groups.csv
    demo_data/test_cases/repeat_metrics_groups.csv
    demo_data/test_cases/mixed_scientific_timeseries.csv

## Что проверяется

| Case | Цель | Основные уязвимости |
|---|---|---|
| `axis_domain_narrow_range` | Проверка осей и использования площади графика | ось от 0 без причины, пустая вертикальная область |
| `compact_boxplot_small_groups` | Проверка компактности категориального графика | широкий холст при 2–4 категориях, пустые горизонтальные промежутки |
| `repeat_axis_labels_metrics` | Проверка repeat/facet подписей | generic `Value`, непонятные названия панелей, нечитаемые подписи |
| `general_orchestrator_scientific_summary` | Проверка общего размыточного запроса | больше 3 графиков, дубли подзадач, неподтверждённые выводы |
| `image_folder_quality` | Проверка image-folder режима | обработка png/jpg/tiff, BRISQUE/NIQE/PIQE, отсутствие удалённых composite-score полей |

## Plan-only запуск

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --cases benchmarks/virage_e2e_test_cases.jsonl --run-id e2e_plan_001

Plan-only режим создаёт `chart_plan.json`, но не запускает генерацию графиков. Это удобно для быстрой проверки AnalysisPlanner.

## Execute запуск

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --cases benchmarks/virage_e2e_test_cases.jsonl --run-id e2e_execute_001 --execute

Execute режим запускает текущий orchestrator и single-chart pipeline для каждой подзадачи.

## Image-folder case

Кейс `image_folder_quality` содержит путь-заглушку:

    {image_folder}

Для запуска этого кейса передай папку с изображениями:

    python scripts/benchmarks/run_virage_e2e_test_cases.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --cases benchmarks/virage_e2e_test_cases.jsonl --run-id e2e_images_001 --case-id image_folder_quality --image-folder path/to/images --execute

Поддерживаемые форматы изображений:

    png jpg jpeg tif tiff bmp gif webp

## Выходные артефакты

Скрипт пишет сводку в:

    artifacts/<run_id>/e2e_cases_report/benchmark_request.json
    artifacts/<run_id>/e2e_cases_report/per_case_results.csv
    artifacts/<run_id>/e2e_cases_report/per_case_results.jsonl
    artifacts/<run_id>/e2e_cases_report/benchmark_summary.json
    artifacts/<run_id>/e2e_cases_report/benchmark_report.md

Каждый orchestrator-run дополнительно сохраняет свои артефакты:

    artifacts/<run_id>/<case_id>/orchestrator_request.json
    artifacts/<run_id>/<case_id>/data_profile.json
    artifacts/<run_id>/<case_id>/chart_plan.json
    artifacts/<run_id>/<case_id>/orchestrator_report.json
    artifacts/<run_id>/<case_id>/final_summary.md

В execute режиме также создаются:

    artifacts/<run_id>/<case_id>/subruns/.../run_report.json

## Что прислать для анализа

После запуска пришли:

    artifacts/<run_id>/e2e_cases_report/benchmark_report.md
    artifacts/<run_id>/e2e_cases_report/per_case_results.csv

Для проблемных кейсов дополнительно:

    chart_plan.json
    orchestrator_report.json
    subruns/*/run_report.json
    subruns/*/nodes/visrag.json
    subruns/*/nodes/visual_chart_judge.json
    chart_spec.vl.json
    chart.png

## Как читать результаты

Сначала смотреть:

- `status` в `per_case_results.csv`;
- число подзадач в `subtask_count`;
- есть ли больше 3 подзадач;
- есть ли skipped/failed cases;
- какие subruns упали.

Для качества графиков смотреть в `run_report.json` и `visual_chart_judge.json`:

- `plot_area_usage_score`;
- `axis_domain_score`;
- `layout_compactness_score`;
- `repeat_axis_label_score`;
- `publication_layout_score`;
- списки `*_issues`.
