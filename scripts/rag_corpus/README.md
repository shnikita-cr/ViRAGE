# Runtime RAG corpus scripts

Основной runtime-путь корпуса:

    python scripts/rag_corpus/loading/download_sources.py --sources wilke_fundamentals from_data_to_viz ft_visual_vocabulary uk_analysis_colours uk_charts_checklist urban_institute_style_guide chartability scientific_figure_guidance image_quality_metrics eda_best_practices

    python scripts/rag_corpus/export_guidance_chunks.py --raw-root rag_corpus/raw_external_rules --output rag_corpus/runtime/guidance_chunks.jsonl --min-chars 220 --max-chars 1000 --overlap-chars 120

    python scripts/rag_corpus/report_corpus_quality.py --input rag_corpus/runtime/guidance_chunks.jsonl --output-dir rag_corpus/reports

    python scripts/rag_corpus/build_runtime_chroma_index.py --chunks rag_corpus/runtime/guidance_chunks.jsonl --persist-dir resources/chroma/virage_guidance_chunks_nomic_embed_text_latest --collection virage_guidance_chunks_nomic_embed_text_latest --embedding-provider ollama --embedding-model nomic-embed-text:latest --embedding-base-url http://localhost:11434 --batch-size 8 --max-input-chars 1600 --recreate

Runtime retrieval modes:

    semantic
    hybrid
    lexical

`semantic` и `hybrid` используют Chroma. `lexical` использует `guidance_chunks.jsonl`.
