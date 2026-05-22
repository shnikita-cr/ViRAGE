from __future__ import annotations

from pathlib import Path

from scripts.rag_corpus.common.io import ensure_dir, project_root, write_text

DEFAULT_OUTPUT = "rag_corpus/autorag/virage_rules/configs/virage_rules_all.yaml"

CONFIG_TEXT = """# AutoRAG all-in-one optimization config for ViRAGE rule/guidance corpus.
# This file intentionally keeps retriever variants in one config so AutoRAG can
# evaluate combinations together and report the best result. Adjust module names
# to the installed AutoRAG version if needed.

node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
        top_k: [1, 2, 3, 5, 8]
        modules:
          - module_type: bm25
          - module_type: vectordb
            embedding_model: [openai, huggingface, ollama]
          - module_type: hybrid_rrf
            weight_range: [[4, 6], [5, 5], [6, 4]]
"""


def export_all_config(output_path: Path | None = None) -> Path:
    root = project_root()
    out = root / (output_path or Path(DEFAULT_OUTPUT))
    ensure_dir(out.parent)
    write_text(out, CONFIG_TEXT)
    return out
