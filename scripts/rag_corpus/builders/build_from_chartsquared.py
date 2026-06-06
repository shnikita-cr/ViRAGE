from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
from pathlib import Path
from typing import Any
try:
    import yaml
except ImportError as exc:
    raise RuntimeError('PyYAML is required to process ChartSquared/ChartUIE yaml files.') from exc

def _load_yaml(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def _extract_query(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ('query', 'instruction', 'initial_instruction', 'user_request', 'request', 'prompt'):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            if isinstance(value, str) and len(value.strip()) > 20:
                return value.strip()
    return ''

def _record_from_uie_file(path: Path, project_root: Path) -> dict[str, Any] | None:
    payload = _load_yaml(path)
    query = _extract_query(payload)
    if not query:
        return None
    rel = path.relative_to(project_root).as_posix()
    return {'record_type': 'chartuie_visual_requirement_seed', 'source_dataset': 'ChartSquared/ChartUIE-8K', 'source_id': path.stem, 'raw_record_path': rel, 'retrieval_text': query, 'user_query': query, 'visual_requirement_seed': {'must_be_visible': [], 'critical_failures': ['Required information is not visible in the static chart image.', 'Requested grouping or highlighting is only available through tooltip or hidden interaction.'], 'yes_no_questions': ['Does the chart image visibly satisfy the user request?', 'Are the requested variables readable in axes, labels, legend, or panels?']}, 'metadata': payload if isinstance(payload, dict) else {}}

def build(project_root: Path, output_path: Path, limit: int | None=None) -> int:
    evaluation_dir = project_root / 'ChartUIE_8K' / 'UIE_evaluation_set'
    if not evaluation_dir.exists():
        raise FileNotFoundError(f'ChartUIE evaluation directory was not found: {evaluation_dir}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open('w', encoding='utf-8') as out:
        for path in sorted(evaluation_dir.glob('*.yaml')):
            record = _record_from_uie_file(path, project_root)
            if record is None:
                continue
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1
            if limit is not None and count >= limit:
                break
    return count

def main() -> None:
    parser = argparse.ArgumentParser(description='Build ViRAGE seed corpus records from ChartSquared ChartUIE-8K.')
    parser.add_argument('--chartsquared-root', required=True, type=Path)
    parser.add_argument('--output', default=Path('rag_corpus/processed/chartsquared_chartuie_seed.jsonl'), type=Path)
    parser.add_argument('--limit', default=None, type=int)
    args = parser.parse_args()
    count = build(args.chartsquared_root, args.output, args.limit)
    logger.info(f'Wrote {count} records to {args.output}')
if __name__ == '__main__':
    main()
