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

Запустить AutoRAG без `autorag` CLI:

    python scripts/rag_corpus/run_autorag_chunks.py evaluate --skip-validation --clean-project-dir --config rag_corpus/autorag/visrag_chunks/configs/visrag_chunks_ollama_all.yaml --qa-data-path rag_corpus/autorag/visrag_chunks/splits/train/qa.parquet --corpus-data-path rag_corpus/autorag/visrag_chunks/splits/train/corpus.parquet --project-dir rag_corpus/autorag/runs/visrag_chunks_train

## Важное правило

LLM-нормализация всего корпуса заранее больше не является основным путём. Runtime VisRAG ищет chunks и генерирует финальный `VisRAGGenerationGuidance` под конкретный `data_profile + query_request_analysis`.
