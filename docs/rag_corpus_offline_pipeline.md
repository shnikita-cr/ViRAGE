# Offline RAG Corpus Pipeline

Документ фиксирует актуальную цепочку подготовки внешнего корпуса правил для ViRAGE.

Основное ограничение: **NLV не используется как источник корпуса**. NLV применяется только для независимой оценки качества построения графиков.

## Назначение корпуса

Runtime RAG не должен хранить готовые Vega-Lite спецификации. Он хранит только правила и подсказки:

    задача пользователя
    структура данных
    тип графика
    кодирование полей
    агрегация / группировка / сортировка
    ограничения читаемости
    антипаттерны

Конкретную Vega-Lite спецификацию всегда создаёт `ChartGeneratorService`.

## Актуальная цепочка

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
    -> runtime export
    -> AutoRAG export

Семантическая дедупликация по эмбеддингам пока запускается вручную:

    python scripts/rag_corpus/normalize/deduplicate_by_embeddings.py --model nomic-embed-text --threshold 0.95

Её результат нужно проверять по `semantic_duplicate_clusters.jsonl`, чтобы не удалить похожие, но разные правила. После этого runtime/AutoRAG export можно запускать с явным профилем корпуса.

Проверка качества корпуса одной командой:

    python scripts/rag_corpus/report_corpus_quality.py --input rag_corpus/processed/all_rules.validated.jsonl

Результаты:

    rag_corpus/reports/corpus_quality_report.json
    rag_corpus/reports/corpus_quality_report.md

## Источники

Используются внешние источники:

    Draco
    From Data to Viz
    Financial Times Visual Vocabulary
    CompassQL
    ChartSquared / C²

TaskVis временно не используется в основном корпусе, потому что в текущей обработке даёт слишком много конкретных примеров.

## Quality filter

Фильтр качества удаляет:

    слишком короткие правила
    технический мусор из документации
    инструкции про библиотеки, установку, тесты и скрипты
    записи с упоминанием NLV
    чрезмерные повторы вроде clear axis labels / plot area / static PNG
    избыток записей от одного источника

Ручной запуск:

    python scripts/rag_corpus/normalize/filter_processed_records.py --input rag_corpus/processed/all_rules.deduped.jsonl --output rag_corpus/processed/all_rules.filtered.jsonl

Полный запуск подготовки:

    python scripts/rag_corpus/run_prepare_corpus.py --provider ollama --model qwen2.5-coder:7b --sources draco from_data_to_viz ft_visual_vocabulary compassql chartsquared_rules --clean-processed

## Runtime RAG

Runtime corpus экспортируется так:

    python scripts/rag_corpus/run_export_runtime.py --profile validated

Для ручной проверки альтернативных профилей:

    python scripts/rag_corpus/run_export_runtime.py --profile filtered
    python scripts/rag_corpus/run_export_runtime.py --profile semantic_deduped

После последних AutoRAG-экспериментов для текущего корпуса наиболее безопасный NLV-режим:

    bm25
    top_k_chart_patterns = 1
    top_k_readability_rules = 0
    top_k_scale_plot_area_rules = 0
    top_k_vlm_readability_rules = 0
    top_k_domain_semantics_rules = 0

Готовый конфиг:

    ui/config/project-gemma4-bench_rag_autorag.toml

Причина: AutoRAG на train показал, что `top_k > 1` быстро добавляет шум. Это признак того, что корпус ещё содержит похожие и неравноценные правила.

## Compatibility reranker

После retrieval действует дополнительный runtime-фильтр совместимости. Он удаляет правила, которые противоречат анализу запроса или требуют отсутствующие данные.

Пример:

    запрос: Show average sales over time by region
    analysis: chart family = line, task = trend
    retrieved rule: choropleth map
    action: reject, если нет географических полей

Это нужно, потому что BM25 может цепляться за слово `region` и доставать географические карты, хотя в задаче `region` является обычной категорией.

## AutoRAG

Экспорт с train/test split:

    python scripts/rag_corpus/run_export_autorag.py --profile validated --train-ratio 0.7 --split-seed 42

Для проверки ручной эмбеддинг-дедупликации:

    python scripts/rag_corpus/run_export_autorag.py --profile semantic_deduped --train-ratio 0.7 --split-seed 42

Validate на train:

    autorag validate --config rag_corpus/autorag/virage_rules/configs/virage_rules_ollama_all.yaml --qa_data_path rag_corpus/autorag/virage_rules/splits/train/qa.parquet --corpus_data_path rag_corpus/autorag/virage_rules/splits/train/corpus.parquet

Evaluate на train:

    autorag evaluate --config rag_corpus/autorag/virage_rules/configs/virage_rules_ollama_all.yaml --qa_data_path rag_corpus/autorag/virage_rules/splits/train/qa.parquet --corpus_data_path rag_corpus/autorag/virage_rules/splits/train/corpus.parquet --project_dir rag_corpus/autorag/runs/ollama_all_train

Извлечь лучшую конфигурацию:

    autorag extract_best_config --trial_path rag_corpus/autorag/runs/ollama_all_train/0 --output_path rag_corpus/autorag/runs/ollama_all_best_config.yaml

Evaluate на test:

    autorag evaluate --config rag_corpus/autorag/runs/ollama_all_best_config.yaml --qa_data_path rag_corpus/autorag/virage_rules/splits/test/qa.parquet --corpus_data_path rag_corpus/autorag/virage_rules/splits/test/corpus.parquet --project_dir rag_corpus/autorag/runs/ollama_all_test

Сводная таблица:

    python scripts/rag_corpus/collect_autorag_summary.py --runs-root rag_corpus/autorag/runs --output-dir rag_corpus/autorag/runs/summary

## Benchmark protocol

NLV без RAG:

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases ./datasets/nlv_corpus/ --config ui/config/project-gemma4-bench_norag.toml --output-dir artifacts/benchmarks/nlv_no_rag --disable-analytics-tail

NLV с текущим RAG:

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases ./datasets/nlv_corpus/ --config ui/config/project-gemma4-bench_rag.toml --output-dir artifacts/benchmarks/nlv_rag_current --disable-analytics-tail

NLV с AutoRAG/runtime-настройкой:

    python scripts/benchmark/run_vegachat_compatible_benchmark.py --cases ./datasets/nlv_corpus/ --config ui/config/project-gemma4-bench_rag_autorag.toml --output-dir artifacts/benchmarks/nlv_rag_autorag --disable-analytics-tail

Сравнение:

    python scripts/benchmark/compare_runs.py --left artifacts/benchmarks/nlv_no_rag --right artifacts/benchmarks/nlv_rag_autorag --output artifacts/benchmarks/nlv_compare_no_rag_vs_rag_autorag

## Критерии готовности корпуса

Корпус можно считать пригодным к полной NLV-проверке, если:

    NLV отсутствует в источниках корпуса
    нет коротких лозунгов вместо правил
    нет технического мусора из документации
    один источник не доминирует выдачу
    ручной retrieval возвращает правила нужной задачи
    top_k=1 не является единственным способом избежать шума
    AutoRAG test-результат зафиксирован отдельно от train

Пока корпус нужно считать экспериментальным: AutoRAG выбрал `BM25 + top_k=1`, что указывает на необходимость дальнейшей очистки и семантической дедупликации.
