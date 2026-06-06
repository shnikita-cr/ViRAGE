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
DEFAULT_INPUT_DIR = 'rag_corpus/raw/vega_lite_examples'
DEFAULT_OUTPUT = 'rag_corpus/extracted/vega_lite_examples.jsonl'

def _mark_type(spec: dict[str, Any]) -> str | None:
    mark = spec.get('mark')
    if isinstance(mark, str):
        return mark
    if isinstance(mark, dict):
        value = mark.get('type')
        return str(value) if value else None
    if 'layer' in spec:
        return 'layer'
    if 'facet' in spec:
        return 'facet'
    return None

def _encoding_summary(spec: dict[str, Any]) -> list[str]:
    encoding = spec.get('encoding')
    if not isinstance(encoding, dict):
        return []
    parts: list[str] = []
    for channel, definition in encoding.items():
        if isinstance(definition, dict):
            field = definition.get('field')
            dtype = definition.get('type')
            aggregate = definition.get('aggregate')
            time_unit = definition.get('timeUnit')
            detail = ':'.join((str(item) for item in (field, dtype) if item))
            extras = ', '.join((str(item) for item in (aggregate, time_unit) if item))
            if extras:
                detail = f'{detail} ({extras})'
            if detail:
                parts.append(f'{channel}={detail}')
    return parts

def extract_vega_lite_examples(input_dir: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for path in sorted(input_dir.rglob('*.json')):
        try:
            spec = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            continue
        if not isinstance(spec, dict):
            continue
        mark = _mark_type(spec)
        encoding = _encoding_summary(spec)
        title = compact_text(spec.get('title') or spec.get('description') or path.stem)
        text_parts = [f'Title: {title}', f'Chart mark/family: {mark}' if mark else '', f"Encoding roles: {'; '.join(encoding)}" if encoding else '', compact_text(spec.get('description')), 'Use this only as source material for rule extraction, not as a runtime Vega-Lite template.']
        text = compact_text('. '.join((part for part in text_parts if part)), max_chars=4000)
        records.append(SourceRecord(record_id=f'vega_lite__{stable_hash(str(path.relative_to(input_dir)))}', source_dataset='vega_lite_examples', source_path=str(path), source_type='vega_lite_example_summary', title=title, text=text, chart_family=mark, metadata={'file_name': path.name, 'relative_source_path': str(path.relative_to(input_dir)), 'encoding_summary': encoding, 'contains_spec_in_raw_only': True}, raw={'title': title, 'mark': mark, 'encoding_summary': encoding}))
    return records

def main() -> None:
    parser = argparse.ArgumentParser(description='Extract Vega-Lite examples into non-spec source summaries for LLM normalization.')
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = project_root()
    records = extract_vega_lite_examples(root / args.input_dir)
    write_jsonl(root / args.output, [record.model_dump() for record in records])
    logger.info(f'Wrote {len(records)} source records to {args.output}')
if __name__ == '__main__':
    main()
