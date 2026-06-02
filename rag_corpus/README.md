# ViRAGE RAG Corpus

Корпус хранит не готовые спецификации графиков, а правила качества визуализации: выбор типа графика, читаемость, подписи, легенды, доступность и текстовое описание графика.

## Структура

- `raw/` — внутренние источники, например обратная связь ViRAGE.
- `raw_external_rules/` — внешние источники правил.
- `extracted/` — извлечённые исходные записи JSONL.
- `processed/` — нормализованные правила после LLM-обработки.
- `runtime/` — компактный JSONL для приложения.
- `autorag/` — конфиги, QA JSONL, datasets, trials и отчёты AutoRAG.
- `reports/` — отчёты качества корпуса.

## Основные источники

- `scientific_figure_guidance`

- `ft_visual_vocabulary`
- `from_data_to_viz`
- `data_visualisation_catalogue`
- `ibm_carbon_chart_anatomy`
- `ibm_carbon_legends`
- `uswds_data_visualizations`
- `urban_institute_style_guide`
- `w3c_wai_complex_images`
- `vistext`

Старые источники `draco`, `compassql`, `chartsquared`, `taskvis`, `vega_lite_examples` больше не входят в основной корпус.

## Типы записей

- `chart_pattern` — какой тип графика подходит задаче.
- `readability_rule` — как сделать график читаемым.
- `scale_plot_area_rule` — как работать с масштабом и областью построения.
- `vlm_readability_rule` — что должно быть видно визуальной модели и человеку.
- `domain_semantics_rule` — дополнительная семантика предметной области.

## Основные команды

Подготовить корпус:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --clean-processed

Проверить качество:

    python scripts/rag_corpus/report_corpus_quality.py --input rag_corpus/processed/all_rules.validated.jsonl

Удалить смысловые повторы:

    python scripts/rag_corpus/normalize/deduplicate_by_embeddings.py --model nomic-embed-text --threshold 0.95

Экспортировать для приложения:

    python scripts/rag_corpus/run_export_runtime.py --profile embedding_deduped

Экспортировать для AutoRAG:

    python scripts/rag_corpus/run_export_autorag.py --profile embedding_deduped --train-ratio 0.7 --split-seed 42

## Политика корпуса

Runtime-документы не должны содержать Vega-Lite `mark`, `encoding`, `$schema`, `spec_template` или готовые спецификации. Корпус должен давать правила и проверки, а не шаблоны для копирования.

## Примечание по извлечению источников качества графиков

Пайплайн дополнительно сохраняет несколько прямых HTML-страниц From Data to Viz, IBM Carbon и USWDS. Это нужно, чтобы извлечение не зависело только от текущей структуры репозиториев. VisText обрабатывается только как структурированный набор подписей и таблиц; файлы метрик, предсказаний и результатов моделей исключаются.


## Runtime RAG strict mode

Runtime RAG работает строго через `guidance_chunks.jsonl` и `guidance_chunk_embeddings.jsonl`. Скрытый lexical fallback запрещён; BM25 используется только как отдельный baseline в AutoRAG.
