# ViRAGE: внешний корпус правил, очистка, AutoRAG и benchmark

Команды запускать из корня проекта:

    D:\programming\projects\ViRAGE

NLV используется только для оценки качества. Запросы, эталонные спецификации и правила, извлечённые из NLV, не добавляются в RAG-корпус.

## 1. Проверка окружения

    python -Wdefault -m compileall -q src ui scripts tests
    pytest -q

    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    pip install pandas pyarrow requests pydantic pyyaml AutoRAG

    ollama list
    python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json().keys())"

Для LLM-нормализации и эмбеддингов:

    ollama pull qwen2.5-coder:7b
    ollama pull nomic-embed-text
    ollama pull mxbai-embed-large
    ollama pull bge-m3


## 1.1. Запуск полной цепочки скриптом

Рядом с этим файлом лежит PowerShell-скрипт, который повторяет основную последовательность команд из `howto.md`:

    .\rag_corpus\run_rag_corpus_pipeline.ps1

Полный запуск с параметрами по умолчанию:

    .\rag_corpus\run_rag_corpus_pipeline.ps1 -LlmModel qwen2.5-coder:7b -EmbeddingModel nomic-embed-text -EmbeddingThreshold 0.95 -ExportProfile embedding_deduped

Если источники и NLV уже скачаны:

    .\rag_corpus\run_rag_corpus_pipeline.ps1 -SkipDownload

Если нужно выполнить только корпус, runtime export и AutoRAG-экспорт без AutoRAG/benchmark:

    .\rag_corpus\run_rag_corpus_pipeline.ps1 -SkipDownload -SkipAutorag -SkipRuntimeConfigApply -SkipNlvSmoke -SkipInfiAgentSmoke

Основной текст ниже оставлен как пошаговая версия тех же команд.

## 2. Скачать внешние источники корпуса

    New-Item -ItemType Directory -Force rag_corpus\raw_external_rules

    git clone https://github.com/uwdata/draco.git rag_corpus\raw_external_rules\draco
    git clone https://github.com/holtzy/data_to_viz.git rag_corpus\raw_external_rules\from_data_to_viz
    git clone https://github.com/Financial-Times/chart-doctor.git rag_corpus\raw_external_rules\ft_visual_vocabulary
    git clone https://github.com/vega/compassql.git rag_corpus\raw_external_rules\compassql
    git clone https://github.com/chartsquared/C-2.git rag_corpus\raw_external_rules\chartsquared

TaskVis сейчас не используется в основном корпусе, потому что он даёт слишком много конкретных примеров. При необходимости его можно скачать отдельно:

    git clone https://github.com/ShenLeixian/TaskVis.git rag_corpus\raw_external_rules\taskvis

Проверка:

    Test-Path rag_corpus\raw_external_rules\draco
    Test-Path rag_corpus\raw_external_rules\from_data_to_viz
    Test-Path rag_corpus\raw_external_rules\ft_visual_vocabulary
    Test-Path rag_corpus\raw_external_rules\compassql
    Test-Path rag_corpus\raw_external_rules\chartsquared

## 3. Скачать NLV только для benchmark

    New-Item -ItemType Directory -Force datasets
    git clone https://github.com/giahy2507/nlvcorpus.github.io.git .\datasets\nlv_corpus
    Invoke-WebRequest "https://docs.google.com/spreadsheets/d/1GMWktNGJCwC8U1dvT0gMggVRRYqN3uL28zjVDbxYJOg/export?format=csv&gid=0" -OutFile .\datasets\nlv_corpus\NLV_Corpus.csv

    Test-Path .\datasets\nlv_corpus\NLV_Corpus.csv
    Test-Path .\datasets\nlv_corpus\vlSpecs.json
    Test-Path .\datasets\nlv_corpus\datasets

## 4. Очистить старые результаты RAG

    Remove-Item -Recurse -Force rag_corpus\extracted -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\processed\llm_normalized -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.deduped.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.filtered.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.embedding_deduped.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\all_rules.validated.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Force rag_corpus\processed\normalization_failures.jsonl -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\runtime -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\autorag\virage_rules -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force rag_corpus\autorag\runs -ErrorAction SilentlyContinue

    New-Item -ItemType Directory -Force rag_corpus\extracted
    New-Item -ItemType Directory -Force rag_corpus\processed\llm_normalized
    New-Item -ItemType Directory -Force rag_corpus\runtime
    New-Item -ItemType Directory -Force rag_corpus\autorag\virage_rules

## 5. Извлечь внешние источники до LLM-нормализации

Извлекатели не должны обходить весь репозиторий источника без отбора. Для каждого источника явно задаются допустимые части пути через `--include-path`, а системные, тестовые и программные области дополнительно отсекаются через `--exclude-path`.

Draco:

    python scripts\rag_corpus\sources\extract_draco.py --include-path docs --include-path constraint --include-path constraints --include-path asp --include-path rules --include-path README.md --exclude-path tests --exclude-path examples --exclude-path node_modules --exclude-path .git

From Data to Viz:

    python scripts\rag_corpus\sources\extract_from_data_to_viz.py --include-path .rmd --include-path readme --include-path caveat --include-path mistake --include-path story --include-path input --exclude-path _site --exclude-path assets --exclude-path static --exclude-path node_modules --exclude-path .git

Financial Times Visual Vocabulary:

    python scripts\rag_corpus\sources\extract_ft_visual_vocabulary.py --include-path visual-vocabulary --include-path README.md --exclude-path node_modules --exclude-path .git

CompassQL:

    python scripts\rag_corpus\sources\extract_compassql.py --include-path README.md --include-path docs --include-path src/rank --include-path src/constraint --include-path src/encoding --include-path src/query --exclude-path test --exclude-path examples --exclude-path website --exclude-path node_modules --exclude-path .git

ChartSquared / C²:

    python scripts\rag_corpus\sources\extract_chartsquared_rules.py --include-path prompt --include-path prompts --include-path criteria --include-path feedback --include-path evaluation --include-path chartaf --include-path chartuie --include-path README.md --exclude-path images --exclude-path assets --exclude-path node_modules --exclude-path .git

Проверить количество исходных записей:

    python -c "from pathlib import Path; [print(p.name, sum(1 for _ in p.open(encoding='utf-8'))) for p in Path('rag_corpus/extracted').glob('*.jsonl')]"

Если какой-то источник дал 0 записей, сначала проверь реальные пути внутри скачанного репозитория:

    Get-ChildItem rag_corpus\raw_external_rules\<source> -Recurse -File | Select-Object -First 50 FullName

После ручного извлечения дальнейшая подготовка запускается с `--skip-extraction`, чтобы `run_prepare_corpus.py` не обходил исходные репозитории повторно.

## 6. LLM-нормализация, точная дедупликация, фильтры и валидация

Основной запуск:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources draco from_data_to_viz ft_visual_vocabulary compassql chartsquared_rules --clean-processed --skip-extraction

Продолжить после обрыва:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources draco from_data_to_viz ft_visual_vocabulary compassql chartsquared_rules --resume --skip-extraction

Повторить только упавшие записи:

    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources draco from_data_to_viz ft_visual_vocabulary compassql chartsquared_rules --retry-failed --skip-extraction

После этого должны появиться:

    rag_corpus\processed\all_rules.jsonl
    rag_corpus\processed\all_rules.deduped.jsonl
    rag_corpus\processed\all_rules.filtered.jsonl
    rag_corpus\processed\all_rules.validated.jsonl
    rag_corpus\processed\processing_report.md

Открыть отчёт:

    notepad rag_corpus\processed\processing_report.md

Сформировать отчёт качества корпуса:

    python scripts\rag_corpus\report_corpus_quality.py --input rag_corpus\processed\all_rules.validated.jsonl
    notepad rag_corpus\reports\corpus_quality_report.md

Проверить отсутствие NLV в корпусе:

    python -c "from pathlib import Path; text=Path('rag_corpus/processed/all_rules.validated.jsonl').read_text(encoding='utf-8').lower(); print('nlv' in text, 'nlv_corpus' in text, 'utterance' in text)"

## 7. Embedding-дедупликация

Этот шаг удаляет смысловые повторы после LLM-нормализации. Он не заменяет точную дедупликацию, а дополняет её за счёт эмбеддингов.

Сначала проверить, что эмбеддинги работают на контрольных примерах:

    python scripts\rag_corpus\normalize\test_embedding_dedup.py --model nomic-embed-text

Открыть отчёт:

    notepad rag_corpus\processed\embedding_dedup_test\embedding_dedup_test_report.md

В отчёте должны быть:

    similar_line_rules: высокая близость
    line_vs_scatter: ниже
    line_vs_bar: ниже
    scatter_vs_bar: ниже

Скрипт перебирает пороги и показывает, при каком пороге похожие правила удаляются, а разные остаются. Если `recommended_threshold` пустой, модель или пороги не подходят для этого корпуса.

Запуск embedding-дедупликации на корпусе:

    python scripts\rag_corpus\normalize\deduplicate_by_embeddings.py --model nomic-embed-text --threshold 0.95

Если удаляются похожие, но разные правила, поднять порог:

    python scripts\rag_corpus\normalize\deduplicate_by_embeddings.py --model nomic-embed-text --threshold 0.97

Если остаётся много повторов, снизить порог:

    python scripts\rag_corpus\normalize\deduplicate_by_embeddings.py --model nomic-embed-text --threshold 0.93

Выходные файлы:

    rag_corpus\processed\all_rules.embedding_deduped.jsonl
    rag_corpus\processed\embedding_duplicate_clusters.jsonl
    rag_corpus\processed\embedding_dedup_report.json
    rag_corpus\processed\embedding_dedup_skipped.jsonl
    rag_corpus\processed\embedding_cache.jsonl

Проверить удалённые кластеры:

    notepad rag_corpus\processed\embedding_duplicate_clusters.jsonl
    notepad rag_corpus\processed\embedding_dedup_report.json

После embedding-дедупликации нужно снова сформировать отчёт качества:

    python scripts\rag_corpus\report_corpus_quality.py --input rag_corpus\processed\all_rules.embedding_deduped.jsonl --output rag_corpus\reports\corpus_quality_embedding_deduped_report.md
    notepad rag_corpus\reports\corpus_quality_embedding_deduped_report.md

## 8. Экспорт корпуса в runtime

Из обычного валидированного профиля:

    python scripts\rag_corpus\run_export_runtime.py --profile validated

Из embedding-dedup профиля:

    python scripts\rag_corpus\run_export_runtime.py --profile embedding_deduped

Проверить runtime-корпус:

    Test-Path rag_corpus\runtime\virage_rules.jsonl
    python -c "from pathlib import Path; p=Path('rag_corpus/runtime/virage_rules.jsonl'); print(p.exists(), p.stat().st_size if p.exists() else 0)"

После этого UI и benchmark будут читать актуальный runtime-корпус из:

    rag_corpus\runtime\virage_rules.jsonl

## 9. Экспорт AutoRAG и train/test split

Из обычного валидированного профиля:

    python scripts\rag_corpus\run_export_autorag.py --profile validated --train-ratio 0.7 --split-seed 42

Из embedding-dedup профиля:

    python scripts\rag_corpus\run_export_autorag.py --profile embedding_deduped --train-ratio 0.7 --split-seed 42

Проверить файлы:

    Test-Path rag_corpus\autorag\virage_rules\corpus.parquet
    Test-Path rag_corpus\autorag\virage_rules\qa.parquet
    Test-Path rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml
    Test-Path rag_corpus\autorag\virage_rules\splits\train\corpus.parquet
    Test-Path rag_corpus\autorag\virage_rules\splits\train\qa.parquet
    Test-Path rag_corpus\autorag\virage_rules\splits\test\corpus.parquet
    Test-Path rag_corpus\autorag\virage_rules\splits\test\qa.parquet

Открыть отчёт split-а:

    notepad rag_corpus\autorag\virage_rules\splits\split_report.md

Проверить формат parquet:

    python -c "import pandas as pd; qa=pd.read_parquet('rag_corpus/autorag/virage_rules/splits/train/qa.parquet'); corpus=pd.read_parquet('rag_corpus/autorag/virage_rules/splits/train/corpus.parquet'); print('train qa', qa.shape); print('train corpus', corpus.shape); print(type(qa['retrieval_gt'].iloc[0]), qa['retrieval_gt'].iloc[0]); print(type(corpus['metadata'].iloc[0]), corpus['metadata'].iloc[0])"

## 10. AutoRAG validate/evaluate/extract/evaluate

Конфигурация `virage_rules_ollama_all.yaml` включает:

    lexical_retrieval: BM25
    semantic_retrieval: Chroma + Ollama embeddings
    hybrid_retrieval: HybridRRF / HybridCC

Гибридный блок исправлен: `HybridRRF.weight_range` задаётся как возрастающий диапазон `[4, 80]`. Нельзя использовать пары вроде `[6, 4]`, потому что AutoRAG вычисляет `max - min + 1` и падает на отрицательном числе.

Validate на train:

    autorag validate --config rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\train\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\train\corpus.parquet

Evaluate на train:

    Remove-Item -Recurse -Force rag_corpus\autorag\runs\ollama_all_train -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force rag_corpus\autorag\runs\ollama_all_train

    autorag evaluate --config rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\train\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\train\corpus.parquet --project_dir rag_corpus\autorag\runs\ollama_all_train

Найти trial-папку:

    Get-ChildItem rag_corpus\autorag\runs\ollama_all_train -Directory

Извлечь лучшую конфигурацию:

    autorag extract_best_config --trial_path rag_corpus\autorag\runs\ollama_all_train\0 --output_path rag_corpus\autorag\runs\ollama_all_best_config.yaml

Если trial-папка не `0`, подставить фактический путь из предыдущей команды.

Evaluate на test:

    Remove-Item -Recurse -Force rag_corpus\autorag\runs\ollama_all_test -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force rag_corpus\autorag\runs\ollama_all_test

    autorag evaluate --config rag_corpus\autorag\runs\ollama_all_best_config.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\test\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\test\corpus.parquet --project_dir rag_corpus\autorag\runs\ollama_all_test

Вся AutoRAG-цепочка одной командой:

    .\rag_corpus\autorag\virage_rules\configs\run_autorag_ollama_configs.ps1

Собрать сводку AutoRAG:

    python scripts\rag_corpus\collect_autorag_summary.py --runs-root rag_corpus\autorag\runs --output-dir rag_corpus\autorag\runs\summary
    notepad rag_corpus\autorag\runs\summary\autorag_summary.md

## 11. Применить AutoRAG-настройку в runtime

Если AutoRAG выбрал конфигурацию, создать benchmark runtime-config:

    python scripts\rag_corpus\apply_runtime_retrieval_config.py --base-config ui\config\benchmark\project-gemma4-bench_rag.toml --output-config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml

Проверить:

    Test-Path ui\config\benchmark\project-gemma4-bench_rag_autorag.toml

Для текущего корпуса AutoRAG ранее показывал, что лучший режим — BM25 с `top_k=1`. Поэтому для NLV нужно использовать строгий config, где лишние readability/plot-area/VLM правила не добавляют шум.

## 12. Ручная проверка RAG в Jupyter

    New-Item -ItemType Directory -Force notebooks
    jupyter notebook notebooks\manual_rag_check.ipynb

В блокноте вручную проверить начало цепочки:

    загрузить config
    создать VisRAGService
    вызвать retrieve_generation_guidance для собственного запроса
    посмотреть prompt_text, retrieved rules и rejected rules

Особенно проверить запрос:

    Show average sales over time by region.

Ожидаемо не должны возвращаться карты, если анализ запроса выбрал line/trend и в данных нет геометрии или координат.

## 13. NLV benchmark

NLV без RAG, smoke:

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\benchmark\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag_smoke --limit 20 --disable-analytics-tail

NLV RAG после AutoRAG, smoke:

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\nlv_rag_autorag_smoke --limit 20 --disable-analytics-tail

Сравнение smoke:

    python scripts\benchmark\compare_runs.py --left artifacts\benchmarks\nlv_no_rag_smoke --right artifacts\benchmarks\nlv_rag_autorag_smoke --output artifacts\benchmarks\nlv_compare_autorag_smoke

Полный NLV без RAG:

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\benchmark\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag --disable-analytics-tail

Полный NLV RAG после AutoRAG:

    python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\nlv_rag_autorag --disable-analytics-tail

Сравнение:

    python scripts\benchmark\compare_runs.py --left artifacts\benchmarks\nlv_no_rag --right artifacts\benchmarks\nlv_rag_autorag --output artifacts\benchmarks\nlv_compare_no_rag_vs_rag_autorag

    python scripts\benchmark\nlv_compare.py --no-rag-report artifacts\benchmarks\nlv_no_rag\benchmark_report.json --rag-report artifacts\benchmarks\nlv_rag_autorag\benchmark_report.json --output-dir artifacts\benchmarks\nlv_compare_metrics

## 14. InfiAgent benchmark

Проверка данных:

    python scripts\benchmark\infiagent_scan.py --source-root Datasets\InfiAgent

Smoke:

    python scripts\benchmark\run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\infiagent_rag_autorag_20 --limit 20

Полный запуск:

    python scripts\benchmark\run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\infiagent_rag_autorag

Отчёт:

    python scripts\benchmark\evaluate_infiagent_results.py --output-dir artifacts\benchmarks\infiagent_rag_autorag

## 15. Что считать готовностью

Перед финальным benchmark должны быть готовы:

    Test-Path rag_corpus\runtime\virage_rules.jsonl
    Test-Path ui\config\benchmark\project-gemma4-bench_rag_autorag.toml
    Test-Path .\datasets\nlv_corpus\NLV_Corpus.csv
    Test-Path .\datasets\nlv_corpus\vlSpecs.json
    Test-Path .\datasets\nlv_corpus\datasets

Критерий успеха на NLV:

    mean_spec_score растёт
    improved > degraded
    broken_by_rag не растёт
    empty_chart_rate не растёт
    visualization_error_rate не растёт

Критерий успеха на InfiAgent:

    accepted_chart_rate растёт
    chart_groundedness растёт
    hallucination_risk падает
    correct_rate или partial_or_correct_rate не падает
