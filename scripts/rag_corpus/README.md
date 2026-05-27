# Скрипты корпуса VisRAG

Новая основная схема корпуса:

    raw_external_rules/*.txt
    → guidance_chunks.jsonl
    → guidance_chunk_embeddings.jsonl
    → AutoRAG corpus/qa parquet

## Основные команды

Скачать источники:

    python scripts/rag_corpus/loading/download_sources.py --refresh

Экспортировать chunks:

    python scripts/rag_corpus/export_guidance_chunks.py

Построить embeddings:

    python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model bge-m3:latest --base-url http://localhost:11434

Экспортировать AutoRAG данные:

    python scripts/rag_corpus/run_export_autorag.py --train-ratio 0.7 --split-seed 42

Полный pipeline:

    python rag_corpus/run_rag_corpus_pipeline.py --skip-download

## Важное правило

LLM-нормализация всего корпуса заранее больше не является основным путём. Runtime VisRAG ищет chunks и генерирует финальный `VisRAGGenerationGuidance` под конкретный `data_profile + query_request_analysis`.
