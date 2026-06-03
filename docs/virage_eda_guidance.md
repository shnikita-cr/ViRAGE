# ViRAGE EDA guidance corpus

`eda_guidance` is a separate RAG source kind for exploratory data analysis. It is distinct from `scientific_figure_guidance`.

`scientific_figure_guidance` answers:

    How should a chart be prepared for a scientific article?

`eda_guidance` answers:

    What should be checked during exploratory data analysis?

## Current source

The initial EDA corpus is a manual seed file:

    rag_corpus/manual_sources/eda_guidance/virage_eda_guidance_rules.txt

The loader copies it into:

    rag_corpus/raw_external_rules/eda_guidance/

Then the standard guidance chunk exporter includes it in:

    rag_corpus/runtime/guidance_chunks.jsonl

## Commands

Prepare EDA guidance raw source:

    python scripts/rag_corpus/loading/load_eda_guidance.py --refresh

Or include it in the shared source downloader:

    python scripts/rag_corpus/loading/download_sources.py --sources eda_guidance --refresh

Rebuild runtime guidance chunks:

    python scripts/rag_corpus/export_guidance_chunks.py --max-chars 1000 --overlap-chars 120

Rebuild runtime embeddings:

    python scripts/rag_corpus/build_visrag_embeddings.py --provider ollama --model nomic-embed-text:latest --base-url http://localhost:11434 --batch-size 1 --max-input-chars 1600

## Intended retrieval use

EDA guidance should be retrieved when the orchestrator selects tasks such as:

- `eda_overview`
- `distribution`
- `missingness_analysis`
- `outlier_detection`
- `correlation`
- `temporal_trend`
- `ranking`

It should not replace task planning. The orchestrator still decides what to analyse; RAG provides task-specific guidance for how to analyse it correctly.
