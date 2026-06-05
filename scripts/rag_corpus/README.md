# RAG corpus pipeline

Основной путь корпуса детерминированный:

```text
скачивание источников
    -> экспорт guidance chunks
    -> построение Chroma index
```

## Runtime corpus

```powershell
python scripts/rag_corpus/loading/download_sources.py --sources wilke_fundamentals from_data_to_viz ft_visual_vocabulary uk_analysis_colours uk_charts_checklist urban_institute_style_guide chartability image_quality_metrics eda_best_practices --refresh
python scripts/rag_corpus/export_guidance_chunks.py --raw-root rag_corpus/raw_external_rules --output rag_corpus/runtime/guidance_chunks.jsonl --min-chars 220 --max-chars 1000 --overlap-chars 120
python scripts/rag_corpus/build_runtime_chroma_index.py --chunks rag_corpus/runtime/guidance_chunks.jsonl --persist-dir resources/chroma/virage_guidance_chunks_nomic_embed_text_latest --collection virage_guidance_chunks_nomic_embed_text_latest --embedding-provider ollama --embedding-model nomic-embed-text:latest --embedding-base-url http://localhost:11434 --batch-size 8 --max-input-chars 1600 --recreate
```

## AutoRAG

AutoRAG запускается отдельно для подбора retrieval-компонентов, параметров и конфигураций. Результат AutoRAG можно вручную перенести в TOML-конфиг после анализа отчёта.

## Runtime contract

Runtime использует:

```text
source corpus: guidance_chunks.jsonl
vector index: Chroma
embedding model: nomic-embed-text:latest
retrieval modes: semantic / hybrid / lexical
```
