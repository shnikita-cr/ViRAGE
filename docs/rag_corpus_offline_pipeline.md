# Offline RAG Corpus Pipeline

Документ фиксирует текущую цепочку подготовки корпуса правил качества графиков для ViRAGE.

## Главный принцип

Корпус не хранит готовые Vega-Lite-спецификации. Он хранит правила:

    выбор типа графика
    читаемость
    подписи и легенды
    доступность
    текстовое описание графика
    проверки для визуальной модели и человека

NLV используется только для оценки. Его нельзя добавлять в RAG-корпус.

## Основные источники

    ft_visual_vocabulary
    from_data_to_viz
    data_visualisation_catalogue
    ibm_carbon_chart_anatomy
    ibm_carbon_legends
    uswds_data_visualizations
    urban_institute_style_guide
    w3c_wai_complex_images
    vistext

Старые источники `draco`, `compassql`, `chartsquared`, `taskvis`, `vega_lite_examples` не входят в основной корпус.

VisText извлекается из структурированных файлов с подписями, табличным представлением или графовой структурой: `data_train.json`, `data_validation.json`, `data_test.json`, а также совместимых JSON/JSONL/CSV/TSV/Parquet-файлов. README и скрипты VisText не используются как записи корпуса.

## Цепочка

    raw_external_rules
    -> extracted/*.jsonl
    -> LLM normalization
    -> all_rules.jsonl
    -> exact deduplication
    -> all_rules.deduped.jsonl
    -> quality filter
    -> all_rules.filtered.jsonl
    -> validation
    -> all_rules.validated.jsonl
    -> embedding deduplication, optional
    -> runtime export
    -> AutoRAG export

## Подготовка корпуса

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --clean-processed

С внутренними правилами и обратной связью ViRAGE:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources ft_visual_vocabulary from_data_to_viz data_visualisation_catalogue ibm_carbon_chart_anatomy ibm_carbon_legends uswds_data_visualizations urban_institute_style_guide w3c_wai_complex_images vistext manual_rules virage_feedback --clean-processed

## Проверка качества

    python scripts/rag_corpus/report_corpus_quality.py --input rag_corpus/processed/all_rules.validated.jsonl

Проверка, что старые источники не попали в корпус:

    python -c "from pathlib import Path; text=Path('rag_corpus/processed/all_rules.validated.jsonl').read_text(encoding='utf-8').lower(); print({x: x in text for x in ['draco','compassql','chartsquared','taskvis','vega_lite_examples','nlv']})"

## Смысловая дедупликация

    python scripts/rag_corpus/normalize/test_embedding_dedup.py --model nomic-embed-text
    python scripts/rag_corpus/normalize/deduplicate_by_embeddings.py --model nomic-embed-text --threshold 0.95

## Runtime export

    python scripts/rag_corpus/run_export_runtime.py --profile validated

После смысловой дедупликации:

    python scripts/rag_corpus/run_export_runtime.py --profile embedding_deduped

## AutoRAG export

    python scripts/rag_corpus/run_export_autorag.py --profile embedding_deduped --train-ratio 0.7 --split-seed 42

## AutoRAG

    autorag validate --config rag_corpus/autorag/virage_rules/configs/virage_rules_ollama_all.yaml --qa_data_path rag_corpus/autorag/virage_rules/splits/train/qa.parquet --corpus_data_path rag_corpus/autorag/virage_rules/splits/train/corpus.parquet

    autorag evaluate --config rag_corpus/autorag/virage_rules/configs/virage_rules_ollama_all.yaml --qa_data_path rag_corpus/autorag/virage_rules/splits/train/qa.parquet --corpus_data_path rag_corpus/autorag/virage_rules/splits/train/corpus.parquet --project_dir rag_corpus/autorag/runs/ollama_all_train

    autorag extract_best_config --trial_path rag_corpus/autorag/runs/ollama_all_train/0 --output_path rag_corpus/autorag/runs/ollama_all_best_config.yaml

    autorag evaluate --config rag_corpus/autorag/runs/ollama_all_best_config.yaml --qa_data_path rag_corpus/autorag/virage_rules/splits/test/qa.parquet --corpus_data_path rag_corpus/autorag/virage_rules/splits/test/corpus.parquet --project_dir rag_corpus/autorag/runs/ollama_all_test

## Оценка

NLV без корпуса:

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases ./datasets/nlv_corpus/ --config ui/config/benchmark/project-gemma4-bench_norag.toml --output-dir artifacts/benchmarks/nlv_no_rag --disable-analytics-tail

NLV с корпусом:

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases ./datasets/nlv_corpus/ --config ui/config/benchmark/project-gemma4-bench_rag_autorag.toml --output-dir artifacts/benchmarks/nlv_rag_autorag --disable-analytics-tail

Сравнение:

    python scripts/benchmark/compare_runs.py --left artifacts/benchmarks/nlv_no_rag --right artifacts/benchmarks/nlv_rag_autorag --output artifacts/benchmarks/nlv_compare_no_rag_vs_rag_autorag

## Готовность корпуса

Корпус пригоден для полной оценки, если:

    старые источники отсутствуют в all_rules.validated.jsonl
    NLV отсутствует в all_rules.validated.jsonl
    нет коротких лозунгов вместо правил
    один источник не доминирует выдачу
    ручная проверка возвращает правила нужной задачи
    AutoRAG test-результат зафиксирован отдельно от train

Проверить ошибки извлечения:

    notepad rag_corpus\reports\extraction_report.json

## Примечание по извлечению источников качества графиков

Пайплайн дополнительно сохраняет несколько прямых HTML-страниц From Data to Viz, IBM Carbon и USWDS. Это нужно, чтобы извлечение не зависело только от текущей структуры репозиториев. VisText обрабатывается только как структурированный набор подписей и таблиц; файлы метрик, предсказаний и результатов моделей исключаются.

