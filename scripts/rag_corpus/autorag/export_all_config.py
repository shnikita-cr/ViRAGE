from __future__ import annotations

from pathlib import Path

from scripts.rag_corpus.common.io import ensure_dir, project_root, write_text

DEFAULT_OUTPUT = "rag_corpus/autorag/virage_rules/configs/virage_rules_ollama_all.yaml"

CONFIG_TEXT = """# ViRAGE AutoRAG config: all supported Ollama retrieval variants in one config.
# This config follows the newer AutoRAG retrieval node split:
# lexical_retrieval, semantic_retrieval, hybrid_retrieval.
# Pull the embedding models before running semantic or hybrid retrieval:
#   ollama pull nomic-embed-text
#   ollama pull mxbai-embed-large
#   ollama pull bge-m3

vectordb:
  - name: chroma_ollama_nomic_embed_text
    db_type: chroma
    client_type: persistent
    embedding_batch: 8
    embedding_model:
      - type: ollama
        model_name: nomic-embed-text
        base_url: http://localhost:11434
    collection_name: virage_rules_nomic_embed_text
    path: ${PROJECT_DIR}/resources/chroma/nomic_embed_text

  - name: chroma_ollama_mxbai_embed_large
    db_type: chroma
    client_type: persistent
    embedding_batch: 8
    embedding_model:
      - type: ollama
        model_name: mxbai-embed-large
        base_url: http://localhost:11434
    collection_name: virage_rules_mxbai_embed_large
    path: ${PROJECT_DIR}/resources/chroma/mxbai_embed_large

  - name: chroma_ollama_bge_m3
    db_type: chroma
    client_type: persistent
    embedding_batch: 4
    embedding_model:
      - type: ollama
        model_name: bge-m3
        base_url: http://localhost:11434
    collection_name: virage_rules_bge_m3
    path: ${PROJECT_DIR}/resources/chroma/bge_m3

node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: lexical_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
          speed_threshold: 10
        top_k: [1, 2, 3, 5, 8]
        modules:
          - module_type: bm25
            bm25_tokenizer: [porter_stemmer, space]

      - node_type: semantic_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
          speed_threshold: 20
        top_k: [1, 2, 3, 5, 8]
        modules:
          - module_type: vectordb
            vectordb:
              - chroma_ollama_nomic_embed_text
              - chroma_ollama_mxbai_embed_large
              - chroma_ollama_bge_m3

      - node_type: hybrid_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
          speed_threshold: 20
        top_k: [1, 2, 3, 5, 8]
        modules:
          - module_type: hybrid_rrf
            # AutoRAG HybridRRF expects an ascending [min, max] range.
            # Do not use pair-like weights such as [6, 4]: AutoRAG computes
            # max - min + 1 internally and fails when the range is reversed.
            weight_range:
              - [4, 80]
          - module_type: hybrid_cc
            normalize_method: [mm, tmm, z, dbsf]
            weight_range:
              - [0.2, 0.8]
            test_weight_size: [7]
"""


def export_all_config(output_path: Path | None = None) -> Path:
    root = project_root()
    out = root / (output_path or Path(DEFAULT_OUTPUT))
    ensure_dir(out.parent)
    write_text(out, CONFIG_TEXT)
    return out
