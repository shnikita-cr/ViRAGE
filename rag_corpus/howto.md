# VisRAG chunk corpus pipeline

Новый основной режим VisRAG не использует заранее сгенерированные `RagRuleRecord`.

Порядок:

    raw_external_rules/*.txt
    → guidance_chunks.jsonl
    → guidance_chunk_embeddings.jsonl
    → runtime retrieve
    → VisRAGResponse
    → spec generation prompt

## 1. Загрузка источников

    python scripts\rag_corpus\loading\download_sources.py --refresh

## 2. Экспорт chunks

Все источники:

    python scripts\rag_corpus\export_guidance_chunks.py

Один источник:

    python scripts\rag_corpus\export_guidance_chunks.py --sources wilke_fundamentals

Результат:

    rag_corpus\runtime\guidance_chunks.jsonl

## 3. Подготовка embeddings

Ollama:

    python scripts\rag_corpus\build_visrag_embeddings.py --provider ollama --model nomic-embed-text --base-url http://localhost:11434

Ollama Cloud/OpenAI/HuggingFace backend выбирается через аргументы provider/model/base-url. Конкретный backend индекса не фиксируется; после AutoRAG можно добавить FAISS/Chroma/DB adapter через `VisRAGStore`.

Результат:

    rag_corpus\runtime\guidance_chunk_embeddings.jsonl

## 4. Runtime

В runtime VisRAG проверяет наличие:

    rag_corpus\runtime\guidance_chunks.jsonl
    rag_corpus\runtime\guidance_chunk_embeddings.jsonl

Если файла нет или embedding отсутствует для chunk, pipeline падает с командой подготовки данных.
