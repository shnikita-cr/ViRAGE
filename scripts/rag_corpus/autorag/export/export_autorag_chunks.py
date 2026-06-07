from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
import re
from pathlib import Path
from typing import Any
import pandas as pd
_CURRENT = Path(__file__).resolve()
ROOT = next((parent for parent in _CURRENT.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), Path.cwd())
from scripts.rag_corpus.autorag.config.export_all_config import export_all_config
from scripts.rag_corpus.common.io import ensure_dir, read_jsonl, write_json, write_text
from scripts.rag_corpus.autorag.export.split_autorag_train_test import split_autorag_data
DEFAULT_INPUT = 'rag_corpus/runtime/guidance_chunks.jsonl'
DEFAULT_OUTPUT_ROOT = 'rag_corpus/autorag/visrag_chunks'
_KEYWORD_PATTERNS: list[tuple[str, str]] = [('\\b(overplot|overlap|scatter|density|jitter|transparent|alpha)\\b', 'dense scatter plots and overlapping points'), ('\\b(color|colour|palette|hue|sequential|diverging|qualitative|rainbow)\\b', 'chart color, palette, and color-scale choice'), ('\\b(label|legend|title|caption|axis|axes|unit)\\b', 'chart labels, titles, axes, legends, and units'), ('\\b(histogram|density|distribution|bin|outlier|boxplot|violin)\\b', 'distribution charts, bins, density, and outliers'), ('\\b(bar|category|categories|categorical|sort|order|rank|ranking)\\b', 'categorical comparison, ranking, sorting, and many categories'), ('\\b(line|time|trend|series|spaghetti)\\b', 'time-series charts, trends, and too many lines'), ('\\b(accessib|contrast|screen reader|alt text|colour blind|color blind)\\b', 'visualization accessibility and non-color encodings'), ('\\b(area|baseline|zero|proportional|scale|log)\\b', 'axes, baseline, scales, and proportional visual size')]

def load_guidance_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f'Guidance chunks file not found: {path}. Run scripts/rag_corpus/export/export_guidance_chunks.py first.')
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f'Guidance chunks file is empty: {path}')
    return rows

def _contents(row: dict[str, Any]) -> str:
    metadata = row.get('metadata') or {}
    parts = [f"Title: {row.get('title') or ''}", f"Source: {row.get('source_name') or row.get('source_id') or ''}", f"Source kind: {row.get('source_kind') or ''}", f"Source path: {row.get('source_path') or ''}", f"Chunk index: {metadata.get('chunk_index') or ''}", 'Guidance source text:', str(row.get('text') or '')]
    return '\n'.join((part for part in parts if str(part).strip()))

def _topic(row: dict[str, Any]) -> str:
    blob = ' '.join([str(row.get('title') or ''), str(row.get('text') or '')]).lower()
    for pattern, topic in _KEYWORD_PATTERNS:
        if re.search(pattern, blob):
            return topic
    title = str(row.get('title') or 'visualization guidance').strip()
    return title or 'visualization guidance'

def _qa_query(row: dict[str, Any]) -> str:
    topic = _topic(row)
    source = str(row.get('source_name') or row.get('source_id') or 'the visualization corpus').strip()
    title = str(row.get('title') or '').strip()
    if title:
        return f'Which practical visualization guidance from {source} helps with {topic}? Relevant section: {title}.'
    return f'Which practical visualization guidance from {source} helps with {topic}?'

def _corpus_rows(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in chunks:
        chunk_id = str(row.get('chunk_id') or '').strip()
        if not chunk_id:
            raise RuntimeError(f'Guidance chunk has no chunk_id: {row}')
        if chunk_id in seen:
            raise RuntimeError(f'Duplicate guidance chunk_id: {chunk_id}')
        seen.add(chunk_id)
        metadata = dict(row.get('metadata') or {})
        metadata.update({'record_type': str(row.get('source_kind') or 'web_guidance'), 'source_kind': str(row.get('source_kind') or 'web_guidance'), 'source_id': str(row.get('source_id') or ''), 'source_name': str(row.get('source_name') or ''), 'source_path': row.get('source_path'), 'url': row.get('url'), 'title': str(row.get('title') or '')})
        rows.append({'doc_id': chunk_id, 'contents': _contents(row), 'metadata': metadata})
    return rows

def _qa_rows(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in chunks:
        chunk_id = str(row['chunk_id'])
        text = str(row.get('text') or '')
        rows.append({'qid': f'qa__chunk__{chunk_id}', 'query': _qa_query(row), 'generation_gt': text[:1200], 'retrieval_gt': [chunk_id], 'metadata': {'source_doc_id': chunk_id, 'source_kind': row.get('source_kind'), 'source_id': row.get('source_id'), 'title': row.get('title')}})
    return rows

def export_autorag_chunks(*, input_path: Path, output_root: Path, train_ratio: float, split_seed: int, no_split: bool=False) -> dict[str, Any]:
    chunks = load_guidance_chunks(input_path)
    ensure_dir(output_root)
    corpus_path = output_root / 'corpus.parquet'
    qa_path = output_root / 'qa.parquet'
    corpus_rows = _corpus_rows(chunks)
    qa_rows = _qa_rows(chunks)
    pd.DataFrame(corpus_rows).to_parquet(corpus_path, index=False)
    pd.DataFrame(qa_rows).to_parquet(qa_path, index=False)
    config_path = export_all_config(Path(output_root.as_posix()) / 'configs' / 'visrag_chunks_ollama_all.yaml')
    split_report = None
    if not no_split:
        split_report = split_autorag_data(corpus_path=corpus_path, qa_path=qa_path, output_root=output_root / 'splits', train_ratio=train_ratio, seed=split_seed)
    report = {'input': str(input_path), 'corpus': {'records': len(corpus_rows), 'output': str(corpus_path)}, 'qa': {'questions': len(qa_rows), 'output': str(qa_path)}, 'config': str(config_path), 'split': split_report}
    write_json(output_root / 'export_report.json', report)
    write_text(output_root / 'export_report.md', '\n'.join(['# VisRAG chunks AutoRAG export', '', f'Input: `{input_path}`', f'Corpus records: **{len(corpus_rows)}**', f'QA questions: **{len(qa_rows)}**', f'Corpus: `{corpus_path}`', f'QA: `{qa_path}`', f'Config: `{config_path}`']))
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description='Export VisRAG guidance chunks to AutoRAG corpus/qa parquet files.')
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--no-split', action='store_true')
    args = parser.parse_args()
    report = export_autorag_chunks(input_path=ROOT / args.input, output_root=ROOT / args.output_root, train_ratio=args.train_ratio, split_seed=args.split_seed, no_split=args.no_split)
    logger.info(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
