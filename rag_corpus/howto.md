# ViRAGE VisRAG chunk corpus pipeline

Команды выполнять из корня проекта.

## 1. Основная схема

    HTML / txt sources
    → clean txt
    → guidance_chunks.jsonl
    → guidance_chunk_embeddings.jsonl
    → runtime VisRAG retrieval
    → VisRAGGenerationGuidance

Старый путь `processed/all_rules.* → RagRuleRecord` больше не является основным режимом.

## 2. Скачать источники

Все источники:

    python scripts\rag_corpus\loading\download_sources.py --refresh

Один источник:

    python scripts\rag_corpus\loading\load_wilke_fundamentals.py --refresh
    python scripts\rag_corpus\loading\load_from_data_to_viz.py --refresh --timeout-seconds 90
    python scripts\rag_corpus\loading\load_ft_visual_vocabulary.py --refresh
    python scripts\rag_corpus\loading\load_uk_analysis_colours.py --refresh
    python scripts\rag_corpus\loading\load_uk_charts_checklist.py --refresh
    python scripts\rag_corpus\loading\load_urban_institute_style_guide.py --refresh
    python scripts\rag_corpus\loading\load_chartability.py --refresh

## 3. Экспорт guidance chunks

    python scripts\rag_corpus\export_guidance_chunks.py

Проверка:

    Get-ChildItem rag_corpus\runtime -File | Select-Object Name, Length

Должен появиться:

    rag_corpus\runtime\guidance_chunks.jsonl

## 4. Подготовка embeddings

Локальная embedding-модель Ollama:

    python scripts\rag_corpus\build_visrag_embeddings.py `
      --provider ollama `
      --model bge-m3:latest `
      --base-url http://localhost:11434

или:

    python scripts\rag_corpus\build_visrag_embeddings.py `
      --provider ollama `
      --model mxbai-embed-large:latest `
      --base-url http://localhost:11434

Проверка:

    Get-ChildItem rag_corpus\runtime -File | Select-Object Name, Length

Должны быть:

    guidance_chunks.jsonl
    guidance_chunk_embeddings.jsonl

Если embeddings отсутствуют, runtime VisRAG падает с ошибкой и командой подготовки.

## 5. Экспорт AutoRAG данных

    python scripts\rag_corpus\run_export_autorag.py --train-ratio 0.7 --split-seed 42

Проверка:

    Test-Path rag_corpus\autorag\visrag_chunks\corpus.parquet
    Test-Path rag_corpus\autorag\visrag_chunks\qa.parquet
    Test-Path rag_corpus\autorag\visrag_chunks\splits\train\corpus.parquet
    Test-Path rag_corpus\autorag\visrag_chunks\splits\train\qa.parquet
    Test-Path rag_corpus\autorag\visrag_chunks\splits\test\corpus.parquet
    Test-Path rag_corpus\autorag\visrag_chunks\splits\test\qa.parquet

## 6. AutoRAG validate/evaluate

Основной запуск AutoRAG выполняется через проектный wrapper, а не через `autorag` CLI. Wrapper импортирует только `autorag.validator.Validator` и `autorag.evaluator.Evaluator`, поэтому не тянет `autorag.deploy`, `gradio` и `fastapi`.

Validate train:

    python scripts\rag_corpus\run_autorag_chunks.py validate `
      --config rag_corpus\autorag\visrag_chunks\configs\visrag_chunks_ollama_all.yaml `
      --qa-data-path rag_corpus\autorag\visrag_chunks\splits\train\qa.parquet `
      --corpus-data-path rag_corpus\autorag\visrag_chunks\splits\train\corpus.parquet

Evaluate train:

    python scripts\rag_corpus\run_autorag_chunks.py evaluate `
      --skip-validation `
      --clean-project-dir `
      --config rag_corpus\autorag\visrag_chunks\configs\visrag_chunks_ollama_all.yaml `
      --qa-data-path rag_corpus\autorag\visrag_chunks\splits\train\qa.parquet `
      --corpus-data-path rag_corpus\autorag\visrag_chunks\splits\train\corpus.parquet `
      --project-dir rag_corpus\autorag\runs\visrag_chunks_train

## 7. Полный запуск

Без AutoRAG evaluate, только подготовка данных:

    python rag_corpus\run_rag_corpus_pipeline.py --skip-download --skip-autorag

Полный запуск с AutoRAG через Python API wrapper:

    python rag_corpus\run_rag_corpus_pipeline.py --skip-download

## 8. Store backend

Runtime использует абстракцию `VisRAGStore`. Сейчас реализован переносимый backend:

    visrag_runtime_store_backend = "jsonl"

FAISS, Chroma или другой backend добавляются как новый adapter без изменения `VisRAGService`.
