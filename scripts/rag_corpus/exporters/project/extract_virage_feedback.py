from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
import argparse
import json
from pathlib import Path
from typing import Any
from scripts.rag_corpus.common.text.hashing import stable_hash
from scripts.rag_corpus.common.io import project_root, write_jsonl
from scripts.rag_corpus.common.models.schemas import SourceRecord
from scripts.rag_corpus.common.text.normalization import compact_text
DEFAULT_INPUT_DIR = 'rag_corpus/raw/virage_feedback'
DEFAULT_OUTPUT = 'rag_corpus/extracted/virage_feedback.jsonl'
TEXT_KEYS = ('feedback', 'judge_feedback', 'user_feedback', 'reason', 'summary', 'error', 'failure_reason', 'visual_observation', 'repair_suggestion', 'query')

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = {'text': line.strip()}
                if isinstance(value, dict):
                    records.append(value)
    return records

def _collect_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in TEXT_KEYS:
        value = record.get(key)
        if value:
            parts.append(f'{key}: {compact_text(value)}')
    if not parts:
        parts.append(compact_text(record, max_chars=2000))
    return compact_text('. '.join(parts), max_chars=4000)

def extract_virage_feedback(input_dir: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for path in sorted(input_dir.rglob('*.jsonl')):
        for idx, item in enumerate(_read_jsonl(path)):
            text = _collect_text(item)
            record_id = str(item.get('record_id') or item.get('id') or f'virage_feedback__{stable_hash([str(path), idx, item])}')
            records.append(SourceRecord(record_id=record_id, source_dataset='virage_feedback', source_path=str(path), source_type='visual_or_user_feedback', title=compact_text(item.get('title') or item.get('query') or 'ViRAGE feedback'), text=text, task=item.get('analysis_task') or item.get('task'), chart_family=item.get('chart_family') or item.get('chart_type'), metadata={'feedback_kind': item.get('feedback_kind') or item.get('source')}, raw=item))
    return records

def main() -> None:
    parser = argparse.ArgumentParser(description='Extract ViRAGE feedback into source records for LLM normalization.')
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = project_root()
    records = extract_virage_feedback(root / args.input_dir)
    write_jsonl(root / args.output, [record.model_dump() for record in records])
    logger.info(f'Wrote {len(records)} source records to {args.output}')
if __name__ == '__main__':
    main()
