# Скрипты корпуса VisRAG

Новая основная схема корпуса:

    raw_external_rules/*.txt
    → guidance_chunks.jsonl
    → guidance_chunk_embeddings.jsonl
    → AutoRAG corpus/qa parquet

## Основные команды

Скачать источники, включая `scientific_figure_guidance`:

    python scripts/rag_corpus/loading/download_sources.py --refresh

Экспортировать chunks:

    python scripts/rag_corpus/export_guidance_chunks.py --max-chars 1000 --overlap-chars 120

Построить embeddings:

    python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model nomic-embed-text:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600

Экспортировать AutoRAG данные:

    python scripts/rag_corpus/run_export_autorag.py --train-ratio 0.7 --split-seed 42

Полный pipeline:

    python rag_corpus/run_rag_corpus_pipeline.py --skip-download

Запустить AutoRAG через project wrapper над `autorag` CLI:

    python scripts/rag_corpus/run_autorag_chunks.py evaluate --skip-validation --clean-project-dir --config rag_corpus/autorag/visrag_chunks/configs/visrag_chunks_ollama_all.yaml --qa-data-path rag_corpus/autorag/visrag_chunks/splits/train/qa.parquet --corpus-data-path rag_corpus/autorag/visrag_chunks/splits/train/corpus.parquet --project-dir rag_corpus/autorag/runs/visrag_chunks_train

## Важное правило

LLM-нормализация всего корпуса заранее больше не является основным путём. Runtime VisRAG ищет chunks и генерирует финальный `VisRAGGenerationGuidance` под конкретный `data_profile + query_request_analysis`.

## AutoRAG dependencies

Для текущей схемы нужны обычный `AutoRAG`, YAML config и Ollama API на `localhost:11434`. Не устанавливать `AutoRAG[gpu]` ради этого сценария: он тянет `vllm`, который не нужен для Windows CPU + Ollama API.

    pip install --upgrade AutoRAG fastapi gradio chromadb pyarrow pandas scikit-learn llama-index-embeddings-ollama

## Runtime RAG strict mode

Runtime VisRAG не выполняет скрытую подмену retrieval backend. Перед запуском `semantic`/`hybrid` должны быть пересобраны `rag_corpus/runtime/guidance_chunks.jsonl` и `rag_corpus/runtime/guidance_chunk_embeddings.jsonl`. `lexical_bm25` — явный backend и не требует embeddings.
