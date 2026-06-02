# ViRAGE RAG corpus and AutoRAG workflow

Команды выполнять из корня проекта.

## 1. Актуальная схема

    raw visualization sources
    → rag_corpus/runtime/guidance_chunks.jsonl
    → rag_corpus/runtime/guidance_chunk_embeddings.jsonl
    → runtime VisRAG retrieval
    → rag_corpus/autorag dataset/config/trials/report
    → AutoRAG retrieval evaluation

Актуальный runtime-корпус:

    rag_corpus/runtime/guidance_chunks.jsonl

Исторический корпус ниже не использовать для текущих оценок и pipeline:

    rag_corpus/runtime_pre/virage_rules.jsonl

Все данные AutoRAG должны лежать только здесь:

    rag_corpus/autorag

Не использовать старые пути:

    configs/autorag
    data/autorag_eval
    artifacts/autorag_eval

## 2. Проверка окружения

Проверить Python-код:

    python -m compileall -q src scripts tests

Проверить unit-тесты:

    pytest -q tests/unit

Проверить AutoRAG CLI:

    autorag --help

Проверить Ollama:

    ollama list

Для AutoRAG и parquet-экспорта нужны зависимости:

    pip install --upgrade AutoRAG fastapi gradio chromadb pyarrow pandas scikit-learn llama-index-embeddings-ollama

Для runtime embeddings нужен локальный Ollama server:

    ollama serve

## 3. Подготовка runtime guidance chunks

Если runtime-файл уже есть и источники не менялись, этот шаг можно пропустить.

Скачать или обновить исходники, включая `scientific_figure_guidance`:

    python scripts/rag_corpus/loading/download_sources.py --refresh

### Ручной cache для защищённых страниц Nature, Cell и JCB

Некоторые страницы Nature, Cell и JCB могут открываться в браузере, но при автоматической загрузке отдавать `Client Challenge`, `cookies_not_supported`, требование JavaScript, `403 Forbidden` или ошибку TLS/SSL-соединения. Такие страницы не нужно обходить кодом. Для них используется ручной cache.

Если `load_scientific_figure_guidance.py --refresh` сообщает, что `nature_initial_submission`, `nature_figure_specifications` или `cell_figure_guidelines` требует ручной cache, открой URL в браузере, сохрани страницу как HTML или скопируй полезный текст в один из файлов:

    rag_corpus/manual_sources/scientific_figure_guidance/nature_initial_submission.html

    rag_corpus/manual_sources/scientific_figure_guidance/nature_initial_submission.txt

    rag_corpus/manual_sources/scientific_figure_guidance/nature_figure_specifications.html

    rag_corpus/manual_sources/scientific_figure_guidance/nature_figure_specifications.txt

    rag_corpus/manual_sources/scientific_figure_guidance/cell_figure_guidelines.html

    rag_corpus/manual_sources/scientific_figure_guidance/cell_figure_guidelines.txt

    rag_corpus/manual_sources/scientific_figure_guidance/jcb_figure_video_guidelines.html

    rag_corpus/manual_sources/scientific_figure_guidance/jcb_figure_video_guidelines.txt

После этого повтори загрузку:

    python scripts/rag_corpus/loading/load_scientific_figure_guidance.py --refresh

Ручной cache не является fallback-источником. Loader принимает его только если файл содержит ожидаемые маркеры полезного текста: `figures`, `figure legend`, `high resolution figures`, `300 dpi`, `Arial or Helvetica`, `RGB colour`, `resolution`, `image`, `video` и похожие требования к научным рисункам.

Экспортировать актуальный chunk-корпус:

    python scripts/rag_corpus/export_guidance_chunks.py --max-chars 1000 --overlap-chars 120

Проверить наличие runtime-корпуса:

    python -c "from pathlib import Path; p=Path('rag_corpus/runtime/guidance_chunks.jsonl'); print(p.exists(), p.stat().st_size if p.exists() else 0)"

Ожидаемый файл:

    rag_corpus/runtime/guidance_chunks.jsonl

Ожидаемые отчёты длины chunks:

    rag_corpus/runtime/runtime_export_report.json
    rag_corpus/runtime/runtime_chunk_length_report.json

Проверить, что в runtime-корпусе нет chunks длиннее лимита:

    python -c "import json; r=json.load(open('rag_corpus/runtime/runtime_chunk_length_report.json', encoding='utf-8')); print(r['max_allowed_chars'], r['max_chunk_chars'], r['oversized_chunks'])"

## 4. Подготовка runtime embeddings для ViRAGE

Этот файл используется runtime-поиском ViRAGE, но не передаётся напрямую в AutoRAG:

    rag_corpus/runtime/guidance_chunk_embeddings.jsonl

Основная команда с `bge-m3`:

    python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model bge-m3:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600

Альтернативные embedding-модели из локального списка:

    python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model mxbai-embed-large:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600

    python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model nomic-embed-text:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600

    python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model qwen3-embedding:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600

Проверить наличие embeddings:

    python -c "from pathlib import Path; p=Path('rag_corpus/runtime/guidance_chunk_embeddings.jsonl'); print(p.exists(), p.stat().st_size if p.exists() else 0)"

Проверить фактическую длину строк, которые отправляются в embedding-модель:

    python -c "import json; r=json.load(open('rag_corpus/runtime/embedding_input_length_report.json', encoding='utf-8')); print(r['max_input_chars'], r['max_input_char_length'], r['exceeded_chunks'])"

## 5. Подготовка AutoRAG QA и dataset

Размеченные retrieval-запросы должны лежать здесь:

    rag_corpus/autorag/qa/retrieval_queries.jsonl

Экспортировать актуальный runtime-корпус в AutoRAG parquet:

    python scripts/autorag_eval/export_autorag_dataset.py --corpus rag_corpus/runtime/guidance_chunks.jsonl --queries rag_corpus/autorag/qa/retrieval_queries.jsonl --output-dir rag_corpus/autorag/datasets/runtime

Проверить результат:

    python -c "from pathlib import Path; print(Path('rag_corpus/autorag/datasets/runtime/corpus.parquet').exists()); print(Path('rag_corpus/autorag/datasets/runtime/qa.parquet').exists()); print(Path('rag_corpus/autorag/datasets/runtime/dataset_manifest.json').exists())"

Ожидаемые файлы:

    rag_corpus/autorag/datasets/runtime/corpus.parquet
    rag_corpus/autorag/datasets/runtime/qa.parquet
    rag_corpus/autorag/datasets/runtime/dataset_manifest.json

## 6. Сборка AutoRAG config

Основной config включает BM25, semantic retrieval через Ollama embeddings и hybrid retrieval:

    python scripts/autorag_eval/build_autorag_config.py --output rag_corpus/autorag/configs/virage_retrieval_eval.yaml

Config с ограниченным набором embedding-моделей:

    python scripts/autorag_eval/build_autorag_config.py --output rag_corpus/autorag/configs/virage_retrieval_eval.yaml --embedding-models bge-m3:latest,nomic-embed-text:latest

Lexical-only baseline для отдельного AutoRAG-сравнения, если нужно оценить BM25 отдельно от semantic/hybrid:

    python scripts/autorag_eval/build_autorag_config.py --output rag_corpus/autorag/configs/virage_retrieval_eval_lexical.yaml --lexical-only

Проверить результат:

    python -c "from pathlib import Path; print(Path('rag_corpus/autorag/configs/virage_retrieval_eval.yaml').exists()); print(Path('rag_corpus/autorag/configs/virage_retrieval_eval.manifest.json').exists())"

## 7. AutoRAG evaluate

Основной запуск:

    autorag evaluate --config rag_corpus/autorag/configs/virage_retrieval_eval.yaml --qa_data_path rag_corpus/autorag/datasets/runtime/qa.parquet --corpus_data_path rag_corpus/autorag/datasets/runtime/corpus.parquet --project_dir rag_corpus/autorag/trials

Lexical-only baseline:

    autorag evaluate --config rag_corpus/autorag/configs/virage_retrieval_eval_lexical.yaml --qa_data_path rag_corpus/autorag/datasets/runtime/qa.parquet --corpus_data_path rag_corpus/autorag/datasets/runtime/corpus.parquet --project_dir rag_corpus/autorag/trials_lexical

AutoRAG должен сам считать retrieval-метрики. В проекте ViRAGE не должно быть самописного расчёта Recall, MRR или nDCG.

## 8. Сбор результатов AutoRAG

Собрать результаты основного запуска:

    python scripts/autorag_eval/collect_autorag_results.py --project-dir rag_corpus/autorag/trials --output-dir rag_corpus/autorag/report

Собрать результаты lexical-only запуска:

    python scripts/autorag_eval/collect_autorag_results.py --project-dir rag_corpus/autorag/trials_lexical --output-dir rag_corpus/autorag/report_lexical

Ожидаемые файлы:

    rag_corpus/autorag/report/autorag_summary.csv
    rag_corpus/autorag/report/autorag_summary.json
    rag_corpus/autorag/report/autorag_report.md

## 9. AutoRAG dashboard

Открыть dashboard для основного запуска:

    autorag dashboard --trial_dir rag_corpus/autorag/trials/0

Если AutoRAG создал другую папку trial, заменить `0` на фактическое имя папки.

## 10. Runtime запуск ViRAGE с RAG

Single-run без RAG:

    python scripts/run_pipeline.py --config ui/config/benchmark/project-gemma4-bench_norag.toml --query "Build a scatter plot of sepal length and petal length" --data-path demo_data/Iris.csv --run-id smoke_no_rag

Single-run с RAG:

    python scripts/run_pipeline.py --config ui/config/benchmark/project-gemma4-bench_rag.toml --query "Build a scatter plot of sepal length and petal length" --data-path demo_data/Iris.csv --run-id smoke_rag

Проверить отчёты:

    python -c "from pathlib import Path; print(Path('artifacts/smoke_no_rag/run_report.json').exists()); print(Path('artifacts/smoke_rag/run_report.json').exists())"

## 11. Полная последовательность для текущего runtime-корпуса

Если `rag_corpus/runtime/guidance_chunks.jsonl` уже есть:

    python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model bge-m3:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600

    python scripts/autorag_eval/export_autorag_dataset.py --corpus rag_corpus/runtime/guidance_chunks.jsonl --queries rag_corpus/autorag/qa/retrieval_queries.jsonl --output-dir rag_corpus/autorag/datasets/runtime

    python scripts/autorag_eval/build_autorag_config.py --output rag_corpus/autorag/configs/virage_retrieval_eval.yaml

    autorag evaluate --config rag_corpus/autorag/configs/virage_retrieval_eval.yaml --qa_data_path rag_corpus/autorag/datasets/runtime/qa.parquet --corpus_data_path rag_corpus/autorag/datasets/runtime/corpus.parquet --project_dir rag_corpus/autorag/trials

    python scripts/autorag_eval/collect_autorag_results.py --project-dir rag_corpus/autorag/trials --output-dir rag_corpus/autorag/report

Если `rag_corpus/runtime/guidance_chunks.jsonl` нужно пересобрать:

    python scripts/rag_corpus/loading/download_sources.py --refresh

    python scripts/rag_corpus/export_guidance_chunks.py --max-chars 1000 --overlap-chars 120

    python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model bge-m3:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600

    python scripts/autorag_eval/export_autorag_dataset.py --corpus rag_corpus/runtime/guidance_chunks.jsonl --queries rag_corpus/autorag/qa/retrieval_queries.jsonl --output-dir rag_corpus/autorag/datasets/runtime

    python scripts/autorag_eval/build_autorag_config.py --output rag_corpus/autorag/configs/virage_retrieval_eval.yaml

    autorag evaluate --config rag_corpus/autorag/configs/virage_retrieval_eval.yaml --qa_data_path rag_corpus/autorag/datasets/runtime/qa.parquet --corpus_data_path rag_corpus/autorag/datasets/runtime/corpus.parquet --project_dir rag_corpus/autorag/trials

    python scripts/autorag_eval/collect_autorag_results.py --project-dir rag_corpus/autorag/trials --output-dir rag_corpus/autorag/report

## 12. Что не использовать

Не использовать исторический runtime_pre для текущей оценки:

    rag_corpus/runtime_pre/virage_rules.jsonl

Не использовать старые AutoRAG layouts:

    configs/autorag
    data/autorag_eval
    artifacts/autorag_eval

Не использовать старые самописные RAG-eval скрипты:

    scripts/rag_eval/run_retrieval_eval.py
    scripts/rag_eval/run_corpus_ablation.py
    src/evaluation/rag_metrics.py

Не использовать `AutoRAG[gpu]` и `vllm` для текущего CPU/API-only контура.


## Runtime RAG strict mode

Runtime ViRAG не использует lexical fallback. Если `rag_corpus/runtime/guidance_chunk_embeddings.jsonl` отсутствует или не покрывает все `chunk_id`, pipeline должен остановиться с ошибкой подготовки корпуса. BM25 используется только как явно выбранный baseline внутри AutoRAG, а не как скрытый runtime fallback.

`recommended_chart_family` не используется в `query_request_analysis` и не участвует в RAG retrieval query. Query analysis описывает пользовательский intent, поля, агрегацию и видимые требования; выбор графика выполняется позже на основе data profile, RAG guidance и spec generation.

## AutoRAG embedding policy

AutoRAG semantic and hybrid retrieval must use embedding batch size 1. The generated config writes this value automatically:

    embedding_batch: 1

Do not include `embeddinggemma:latest` in AutoRAG experiments. Use only:

    bge-m3:latest
    mxbai-embed-large:latest
    nomic-embed-text:latest
    qwen3-embedding:latest
