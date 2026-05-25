# ViRAGE: подготовка корпуса правил качества графиков

Команды запускать из корня проекта.

## 1. Назначение корпуса

Корпус теперь строится только из читаемых источников с правилами качества графиков. В основной корпус не входят `Draco`, `CompassQL`, `ChartSquared`, `TaskVis`, `Vega-Lite examples`, `NLV` и другие наборы вида “запрос -> готовый график”.

В корпус добавлены 9 источников:

| id | Источник | Формат |
|---|---|---|
| `ft_visual_vocabulary` | Financial Times Visual Vocabulary | репозиторий с разметкой и текстом |
| `from_data_to_viz` | From Data to Viz | репозиторий с текстовыми страницами |
| `data_visualisation_catalogue` | Data Visualisation Catalogue | сохранённые HTML-страницы |
| `ibm_carbon_chart_anatomy` | IBM Carbon Chart Anatomy | сохранённая HTML-страница |
| `ibm_carbon_legends` | IBM Carbon Legends | сохранённая HTML-страница |
| `uswds_data_visualizations` | USWDS Data Visualizations | сохранённая HTML-страница |
| `urban_institute_style_guide` | Urban Institute Style Guide | сохранённая HTML-страница |
| `w3c_wai_complex_images` | W3C WAI Complex Images | сохранённая HTML-страница |
| `vistext` | VisText | структурированные файлы, текстовые описания, таблицы; изображения не используются |

## 2. Проверка окружения

    python -Wdefault -m compileall -q src ui scripts tests
    pytest -q

    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    pip install pandas pyarrow requests pydantic pyyaml AutoRAG

    ollama list
    ollama pull qwen2.5-coder:7b
    ollama pull nomic-embed-text

## 3. Полный запуск одной командой

    .\rag_corpus\run_rag_corpus_pipeline.ps1 -LlmModel qwen2.5-coder:7b -EmbeddingModel nomic-embed-text -EmbeddingThreshold 0.95 -ExportProfile embedding_deduped

Если источники уже скачаны:

    .\rag_corpus\run_rag_corpus_pipeline.ps1 -SkipDownload

Если нужно только подготовить корпус и экспортировать файлы без оценки:

    .\rag_corpus\run_rag_corpus_pipeline.ps1 -SkipDownload -SkipAutorag -SkipRuntimeConfigApply -SkipNlvSmoke -SkipInfiAgentSmoke

## 4. Скачать источники корпуса вручную

Создать папки:

    New-Item -ItemType Directory -Force rag_corpus\raw_external_rules

Репозитории:

    git clone https://github.com/Financial-Times/chart-doctor.git rag_corpus\raw_external_rules\ft_visual_vocabulary
    git clone https://github.com/holtzy/data_to_viz.git rag_corpus\raw_external_rules\from_data_to_viz
    git clone https://github.com/mitvis/vistext.git rag_corpus\raw_external_rules\vistext

HTML-источники:

    New-Item -ItemType Directory -Force rag_corpus\raw_external_rules\data_visualisation_catalogue
    Invoke-WebRequest https://datavizcatalogue.com/ -OutFile rag_corpus\raw_external_rules\data_visualisation_catalogue\index.html
    Invoke-WebRequest https://datavizcatalogue.com/methods/bar_chart.html -OutFile rag_corpus\raw_external_rules\data_visualisation_catalogue\bar_chart.html
    Invoke-WebRequest https://datavizcatalogue.com/methods/line_graph.html -OutFile rag_corpus\raw_external_rules\data_visualisation_catalogue\line_graph.html
    Invoke-WebRequest https://datavizcatalogue.com/methods/scatterplot.html -OutFile rag_corpus\raw_external_rules\data_visualisation_catalogue\scatterplot.html
    Invoke-WebRequest https://datavizcatalogue.com/methods/histogram.html -OutFile rag_corpus\raw_external_rules\data_visualisation_catalogue\histogram.html
    Invoke-WebRequest https://datavizcatalogue.com/methods/treemap.html -OutFile rag_corpus\raw_external_rules\data_visualisation_catalogue\treemap.html

    New-Item -ItemType Directory -Force rag_corpus\raw_external_rules\ibm_carbon_chart_anatomy
    Invoke-WebRequest https://carbondesignsystem.com/data-visualization/chart-anatomy/ -OutFile rag_corpus\raw_external_rules\ibm_carbon_chart_anatomy\index.html

    New-Item -ItemType Directory -Force rag_corpus\raw_external_rules\ibm_carbon_legends
    Invoke-WebRequest https://carbondesignsystem.com/data-visualization/legends/ -OutFile rag_corpus\raw_external_rules\ibm_carbon_legends\index.html

    New-Item -ItemType Directory -Force rag_corpus\raw_external_rules\uswds_data_visualizations
    Invoke-WebRequest https://designsystem.digital.gov/components/data-visualizations/ -OutFile rag_corpus\raw_external_rules\uswds_data_visualizations\index.html

    New-Item -ItemType Directory -Force rag_corpus\raw_external_rules\urban_institute_style_guide
    Invoke-WebRequest https://urbaninstitute.github.io/graphics-styleguide/ -OutFile rag_corpus\raw_external_rules\urban_institute_style_guide\index.html

    New-Item -ItemType Directory -Force rag_corpus\raw_external_rules\w3c_wai_complex_images
    Invoke-WebRequest https://www.w3.org/WAI/tutorials/images/complex/ -OutFile rag_corpus\raw_external_rules\w3c_wai_complex_images\index.html

Проверка:

    Test-Path rag_corpus\raw_external_rules\ft_visual_vocabulary
    Test-Path rag_corpus\raw_external_rules\from_data_to_viz
    Test-Path rag_corpus\raw_external_rules\vistext
    Test-Path rag_corpus\raw_external_rules\data_visualisation_catalogue\index.html
    Test-Path rag_corpus\raw_external_rules\ibm_carbon_chart_anatomy\index.html
    Test-Path rag_corpus\raw_external_rules\ibm_carbon_legends\index.html
    Test-Path rag_corpus\raw_external_rules\uswds_data_visualizations\index.html
    Test-Path rag_corpus\raw_external_rules\urban_institute_style_guide\index.html
    Test-Path rag_corpus\raw_external_rules\w3c_wai_complex_images\index.html

## 5. Подготовить корпус

Основной запуск по 9 источникам:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --clean-processed

Продолжить после обрыва:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --resume

Повторить только упавшие записи:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --retry-failed

Подключить внутренние правила и обратную связь ViRAGE дополнительно:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources ft_visual_vocabulary from_data_to_viz data_visualisation_catalogue ibm_carbon_chart_anatomy ibm_carbon_legends uswds_data_visualizations urban_institute_style_guide w3c_wai_complex_images vistext manual_rules virage_feedback --clean-processed

После запуска должны появиться:

    rag_corpus\processed\all_rules.jsonl
    rag_corpus\processed\all_rules.deduped.jsonl
    rag_corpus\processed\all_rules.filtered.jsonl
    rag_corpus\processed\all_rules.validated.jsonl
    rag_corpus\processed\processing_report.md

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
