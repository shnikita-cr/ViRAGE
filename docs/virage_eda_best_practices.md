# EDA best-practice corpus

The EDA corpus is loaded from real external sources and exported to runtime RAG as `eda_guidance` chunks.

Sources:

    NIST/SEMATECH e-Handbook of Statistical Methods, Chapter 1: Exploratory Data Analysis
    R for Data Science, Chapter 7: Exploratory Data Analysis

The corpus provides guidance for:

    distribution checks
    missing-value inspection
    outlier and anomaly discovery
    variable relationships
    high-cardinality categories
    iterative refinement of analytical questions
    graphics-first exploratory analysis

Download/cache the source pages:

    python scripts/rag_corpus/loading/load_eda_best_practices.py --refresh

Or include the source in the full practical corpus download:

    python scripts/rag_corpus/loading/download_sources.py --sources eda_best_practices --refresh

Then rebuild runtime chunks and the Chroma index:

    python scripts/rag_corpus/export_guidance_chunks.py --max-chars 1000 --overlap-chars 120

    python scripts/rag_corpus/build_runtime_chroma_index.py --chunks rag_corpus/runtime/guidance_chunks.jsonl --persist-dir resources/chroma/virage_guidance_chunks_nomic_embed_text_latest --collection virage_guidance_chunks_nomic_embed_text_latest --embedding-provider ollama --embedding-model nomic-embed-text:latest --embedding-base-url http://localhost:11434 --recreate
