from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
import re
from pathlib import Path
from typing import Any
_CURRENT = Path(__file__).resolve()
ROOT = next((parent for parent in _CURRENT.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), Path.cwd())
from scripts.rag_corpus.common.io import ensure_dir, write_json, write_text
DEFAULT_OUTPUT = 'rag_corpus/autorag/configs/virage_retrieval_eval.yaml'
DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434'
DEFAULT_OLLAMA_EMBEDDING_MODELS = ['nomic-embed-text:latest']
EXCLUDED_OLLAMA_EMBEDDING_MODELS: set[str] = set()

def _parse_embedding_models(raw_models: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if raw_models is None:
        return list(DEFAULT_OLLAMA_EMBEDDING_MODELS)
    if isinstance(raw_models, str):
        models = [item.strip() for item in raw_models.split(',')]
    else:
        models = [str(item).strip() for item in raw_models]
    return [model for model in models if model and model not in EXCLUDED_OLLAMA_EMBEDDING_MODELS]

def _safe_name(model_name: str) -> str:
    value = re.sub('[^a-zA-Z0-9]+', '_', model_name).strip('_').lower()
    return value or 'embedding_model'

def _vectordb_name(model_name: str) -> str:
    return f'virage_chroma_{_safe_name(model_name)}'

def _vectordb_config(*, embedding_models: list[str], ollama_base_url: str) -> str:
    lines = ['vectordb:']
    for model in embedding_models:
        suffix = _safe_name(model)
        lines.extend([f'  - name: {_vectordb_name(model)}', '    db_type: chroma', '    client_type: persistent', '    embedding_batch: 1', '    embedding_model:', '      - type: ollama', f'        model_name: {model}', f'        base_url: {ollama_base_url}', f'    collection_name: virage_guidance_chunks_{suffix}', f'    path: ${{PROJECT_DIR}}/resources/chroma/virage_guidance_chunks_{suffix}', ''])
    return '\n'.join(lines).rstrip()

def _semantic_modules(embedding_models: list[str]) -> list[str]:
    lines: list[str] = []
    for model in embedding_models:
        lines.extend(['          - module_type: vectordb', f'            vectordb: {_vectordb_name(model)}'])
    return lines

def _retrieve_config(*, embedding_models: list[str], ollama_base_url: str, lexical_only: bool) -> str:
    if lexical_only:
        return '# ViRAGE AutoRAG retrieval evaluation config.\n# Lexical-only baseline. The main config should include lexical, semantic and hybrid retrieval.\n\nnode_lines:\n  - node_line_name: retrieve_node_line\n    nodes:\n      - node_type: lexical_retrieval\n        strategy:\n          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]\n          speed_threshold: 10\n        top_k: [1, 3, 5, 8]\n        modules:\n          - module_type: bm25\n            bm25_tokenizer: [porter_stemmer, space]\n'
    semantic_modules = '\n'.join(_semantic_modules(embedding_models))
    return f'# ViRAGE AutoRAG retrieval evaluation config.\n# AutoRAG computes retrieval metrics and compares lexical, semantic and hybrid retrieval.\n# Data, config, labelled QA and trial outputs must stay under rag_corpus/autorag/.\n# Requires AutoRAG >= 0.3.17 for lexical_retrieval / semantic_retrieval / hybrid_retrieval nodes.\n\n{_vectordb_config(embedding_models=embedding_models, ollama_base_url=ollama_base_url)}\n\nnode_lines:\n  - node_line_name: retrieve_node_line\n    nodes:\n      - node_type: lexical_retrieval\n        strategy:\n          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]\n          speed_threshold: 10\n        top_k: [1, 3, 5, 8]\n        modules:\n          - module_type: bm25\n            bm25_tokenizer: [porter_stemmer, space]\n\n      - node_type: semantic_retrieval\n        strategy:\n          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]\n          speed_threshold: 30\n        top_k: [1, 3, 5, 8]\n        modules:\n{semantic_modules}\n\n      - node_type: hybrid_retrieval\n        strategy:\n          metrics: [retrieval_f1, retrieval_recall, retrieval_precision]\n          speed_threshold: 40\n        top_k: [1, 3, 5, 8]\n        modules:\n          - module_type: hybrid_rrf\n            weight_range: (4, 80)\n          - module_type: hybrid_cc\n            normalize_method: [mm, tmm, z, dbsf]\n            weight_range: (0.0, 1.0)\n            test_weight_size: 11\n'

def build_autorag_config(*, output_path: Path, embedding_models: list[str] | tuple[str, ...] | str | None=None, ollama_base_url: str=DEFAULT_OLLAMA_BASE_URL, lexical_only: bool=False) -> dict[str, Any]:
    resolved_models = _parse_embedding_models(embedding_models)
    ensure_dir(output_path.parent)
    text = _retrieve_config(embedding_models=resolved_models, ollama_base_url=ollama_base_url, lexical_only=lexical_only)
    write_text(output_path, text)
    report = {'output': str(output_path), 'embedding_provider': 'ollama' if not lexical_only else None, 'embedding_models': [] if lexical_only else resolved_models, 'embedding_batch': None if lexical_only else 1, 'excluded_embedding_models': sorted(EXCLUDED_OLLAMA_EMBEDDING_MODELS), 'ollama_base_url': None if lexical_only else ollama_base_url, 'lexical_only': lexical_only, 'retrieval_nodes': ['lexical_retrieval'] if lexical_only else ['lexical_retrieval', 'semantic_retrieval', 'hybrid_retrieval'], 'autorag_root': 'rag_corpus/autorag', 'note': 'AutoRAG computes retrieval metrics during autorag evaluate. ViRAGE does not recalculate them.'}
    write_json(output_path.with_suffix('.manifest.json'), report)
    return report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Write an AutoRAG YAML config for ViRAGE retrieval evaluation.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--embedding-models', default=','.join(DEFAULT_OLLAMA_EMBEDDING_MODELS), help='Comma-separated Ollama embedding model names.')
    parser.add_argument('--ollama-base-url', default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument('--lexical-only', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    report = build_autorag_config(output_path=ROOT / args.output, embedding_models=args.embedding_models, ollama_base_url=args.ollama_base_url, lexical_only=args.lexical_only)
    logger.info(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
