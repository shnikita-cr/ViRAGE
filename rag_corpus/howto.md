# ViRAGE: подготовка корпуса правил качества графиков

Команды запускать из корня проекта.

## 1. Назначение корпуса

Корпус строится для этапа `visrag`: практические правила выбора графика, ошибок, цвета, подписей, читаемости и доступности. В основной корпус не входят `Draco`, `CompassQL`, `ChartSquared`, `TaskVis`, `Vega-Lite examples`, `NLV` и наборы вида “запрос -> готовый график”.

Основные источники:

| id | Источник | Что даёт |
|---|---|---|
| `wilke_fundamentals` | Claus Wilke, Fundamentals of Data Visualization | сравнение, распределения, overplotting, цвет, оси, подписи |
| `from_data_to_viz` | From Data to Viz / Caveats | типовые ошибки и выбор графика по форме данных |
| `ft_visual_vocabulary` | Financial Times Visual Vocabulary | выбор графика по аналитической задаче |
| `uk_analysis_colours` | UK Analysis Function: colours | доступный цвет и тип цветовой шкалы |
| `uk_charts_checklist` | UK Analysis Function: charts checklist | проверки графика перед публикацией |
| `urban_institute_style_guide` | Urban Institute Style Guide | подписи, цвета, layout, аннотации |
| `chartability` | Chartability / POUR-CAF | accessibility-аудит визуализаций |

## 2. Проверка окружения

    python -Wdefault -m compileall -q src ui scripts tests
    pytest -q

    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    pip install pandas pyarrow requests pydantic pyyaml AutoRAG

    # Модели должны быть доступны через Ollama API. Pipeline не вызывает ollama CLI.

## 3. Полный запуск одной командой

    python rag_corpus\run_rag_corpus_pipeline.py --llm-model qwen2.5-coder:7b --embedding-model nomic-embed-text --embedding-threshold 0.95 --export-profile embedding_deduped

Если источники уже скачаны:

    python rag_corpus\run_rag_corpus_pipeline.py --skip-download

Если нужно только подготовить корпус и экспортировать файлы без оценки:

    python rag_corpus\run_rag_corpus_pipeline.py --skip-download --skip-autorag --skip-runtime-config-apply --skip-nlv-smoke --skip-infiagent-smoke

## 4. Скачать источники корпуса

Основная команда:

    python scripts\rag_corpus\loading\download_sources.py

Принудительно обновить сохранённые страницы:

    python scripts\rag_corpus\loading\download_sources.py --refresh

Скачать только выбранные источники:

    python scripts\rag_corpus\loading\download_sources.py --sources wilke_fundamentals from_data_to_viz ft_visual_vocabulary uk_analysis_colours uk_charts_checklist urban_institute_style_guide chartability

Проверка:

    notepad rag_corpus\reports\source_download_report.json
    Get-ChildItem rag_corpus\raw_external_rules -Directory


## Мини-загрузчики источников

Каждый внешний источник можно загрузить отдельно:

    python scripts\rag_corpus\loading\load_wilke_fundamentals.py --refresh
    python scripts\rag_corpus\loading\load_from_data_to_viz.py --refresh
    python scripts\rag_corpus\loading\load_ft_visual_vocabulary.py --refresh
    python scripts\rag_corpus\loading\load_uk_analysis_colours.py --refresh
    python scripts\rag_corpus\loading\load_uk_charts_checklist.py --refresh
    python scripts\rag_corpus\loading\load_urban_institute_style_guide.py --refresh
    python scripts\rag_corpus\loading\load_chartability.py --refresh

Загрузчики используют BeautifulSoup для HTML-страниц, обходят только полезные внутренние HTML-ссылки и не используют fallback-тексты.

## 5. Подготовить корпус

Загрузка и извлечение работают строго: ошибка скачивания, подозрительно маленький файл, отсутствующая raw-папка или нулевое извлечение записей останавливают pipeline. Подстановочные тексты не используются.


Основной запуск:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --clean-processed

Без повторной загрузки источников, только если `rag_corpus\raw_external_rules\*` уже реально загружены:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --clean-processed --skip-source-download

Продолжить после обрыва:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --resume

Повторить только упавшие записи:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --retry-failed

Подключить внутренние правила и обратную связь ViRAGE дополнительно:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources wilke_fundamentals from_data_to_viz ft_visual_vocabulary uk_analysis_colours uk_charts_checklist urban_institute_style_guide chartability manual_rules virage_feedback --clean-processed

После запуска должны появиться:

    rag_corpus\extracted\*.jsonl
    rag_corpus\processed\all_rules.jsonl
    rag_corpus\processed\all_rules.deduped.jsonl
    rag_corpus\processed\all_rules.filtered.jsonl
    rag_corpus\processed\all_rules.validated.jsonl
    rag_corpus\processed\processing_report.md
    rag_corpus\reports\extraction_report.json

Проверить количество извлечённых записей:

    python -c "from pathlib import Path; [print(p.name, sum(1 for _ in p.open(encoding='utf-8'))) for p in Path('rag_corpus/extracted').glob('*.jsonl')]"

Если источник дал 0 записей, сначала посмотреть реальные файлы:

    Get-ChildItem rag_corpus\raw_external_rules\<source> -Recurse -File | Select-Object -First 50 FullName

## 6. Проверить качество корпуса

    python scripts\rag_corpus\report_corpus_quality.py --input rag_corpus\processed\all_rules.validated.jsonl
    notepad rag_corpus\reports\corpus_quality_report.md

Проверить, что старые источники не попали в корпус:

    python -c "from pathlib import Path; text=Path('rag_corpus/processed/all_rules.validated.jsonl').read_text(encoding='utf-8').lower(); print({x: x in text for x in ['draco','compassql','chartsquared','taskvis','vega_lite_examples','nlv']})"

## 7. Смысловая дедупликация

Проверить модель:

    python scripts\rag_corpus\normalize\test_embedding_dedup.py --model nomic-embed-text

Запуск:

    python scripts\rag_corpus\normalize\deduplicate_by_embeddings.py --model nomic-embed-text --threshold 0.95

Если удаляются разные правила, поднять порог:

    python scripts\rag_corpus\normalize\deduplicate_by_embeddings.py --model nomic-embed-text --threshold 0.97

Если остаётся много повторов, снизить порог:

    python scripts\rag_corpus\normalize\deduplicate_by_embeddings.py --model nomic-embed-text --threshold 0.93

## 8. Экспорт для работы приложения

Без смысловой дедупликации:

    python scripts\rag_corpus\run_export_runtime.py --profile validated

После смысловой дедупликации:

    python scripts\rag_corpus\run_export_runtime.py --profile embedding_deduped

Проверка:

    Test-Path rag_corpus\runtime\virage_rules.jsonl
    notepad rag_corpus\runtime\runtime_export_report.json

## 9. Экспорт для AutoRAG

    python scripts\rag_corpus\run_export_autorag.py --profile embedding_deduped --train-ratio 0.7 --split-seed 42

Проверка:

    Test-Path rag_corpus\autorag\virage_rules\corpus.parquet
    Test-Path rag_corpus\autorag\virage_rules\qa.parquet
    Test-Path rag_corpus\autorag\virage_rules\splits\train\corpus.parquet
    Test-Path rag_corpus\autorag\virage_rules\splits\test\qa.parquet

## 10. AutoRAG

Проверка train:

    autorag validate --config rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\train\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\train\corpus.parquet

Подбор на train:

    Remove-Item -Recurse -Force rag_corpus\autorag\runs\ollama_all_train -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force rag_corpus\autorag\runs\ollama_all_train
    autorag evaluate --config rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\train\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\train\corpus.parquet --project_dir rag_corpus\autorag\runs\ollama_all_train

Извлечь лучшую конфигурацию:

    autorag extract_best_config --trial_path rag_corpus\autorag\runs\ollama_all_train\0 --output_path rag_corpus\autorag\runs\ollama_all_best_config.yaml

Проверить на test:

    Remove-Item -Recurse -Force rag_corpus\autorag\runs\ollama_all_test -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force rag_corpus\autorag\runs\ollama_all_test
    autorag evaluate --config rag_corpus\autorag\runs\ollama_all_best_config.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\test\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\test\corpus.parquet --project_dir rag_corpus\autorag\runs\ollama_all_test

Сводка:

    python scripts\rag_corpus\collect_autorag_summary.py --runs-root rag_corpus\autorag\runs --output-dir rag_corpus\autorag\runs\summary

## 11. Применить настройку поиска правил

    python scripts\rag_corpus\apply_runtime_retrieval_config.py --base-config ui\config\benchmark\project-gemma4-bench_rag.toml --output-config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml

## 12. Проверка на NLV

NLV остаётся только внешним набором для проверки. Его нельзя добавлять в корпус.

Скачать:

    New-Item -ItemType Directory -Force datasets
    git clone https://github.com/giahy2507/nlvcorpus.github.io.git .\datasets\nlv_corpus
    Invoke-WebRequest "https://docs.google.com/spreadsheets/d/1GMWktNGJCwC8U1dvT0gMggVRRYqN3uL28zjVDbxYJOg/export?format=csv&gid=0" -OutFile .\datasets\nlv_corpus\NLV_Corpus.csv

Проверка без корпуса:

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\benchmark\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag_smoke --limit 20 --disable-analytics-tail

Проверка с корпусом:

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\nlv_rag_autorag_smoke --limit 20 --disable-analytics-tail

Сравнение:

    python scripts\benchmark\compare_runs.py --left artifacts\benchmarks\nlv_no_rag_smoke --right artifacts\benchmarks\nlv_rag_autorag_smoke --output artifacts\benchmarks\nlv_compare_autorag_smoke

## 13. Проверка на InfiAgent

    python scripts\benchmark\infiagent_scan.py --source-root Datasets\InfiAgent
    python scripts\benchmark\run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\infiagent_rag_autorag_20 --limit 20
    python scripts\benchmark\evaluate_infiagent_results.py --output-dir artifacts\benchmarks\infiagent_rag_autorag_20

## 14. Критерии готовности

Готово, если:

    rag_corpus\runtime\virage_rules.jsonl создан
    rag_corpus\processed\processing_report.md не показывает доминирование одного источника
    старые источники отсутствуют в all_rules.validated.jsonl
    NLV отсутствует в all_rules.validated.jsonl
    ручная проверка возвращает правила выбора графика, читаемости, подписей, легенд и текстового описания
    NLV smoke не показывает рост broken_by_rag и visualization_error_rate

## Примечание по извлечению источников качества графиков

Пайплайн дополнительно сохраняет несколько прямых HTML-страниц From Data to Viz, IBM Carbon и USWDS. Это нужно, чтобы извлечение не зависело только от текущей структуры репозиториев. VisText обрабатывается только как структурированный набор подписей и таблиц; файлы метрик, предсказаний и результатов моделей исключаются.

