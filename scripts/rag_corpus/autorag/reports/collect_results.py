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
import pandas as pd
from scripts.rag_corpus.common.io import project_root, write_json, write_text
DEFAULT_INPUT_DIR = 'rag_corpus/autorag/virage_rules'
DEFAULT_OUT_CSV = 'rag_corpus/autorag/virage_rules/reports/retrieval_comparison.csv'
DEFAULT_OUT_JSON = 'rag_corpus/autorag/virage_rules/reports/retrieval_comparison.json'
DEFAULT_OUT_MD = 'rag_corpus/autorag/virage_rules/reports/retrieval_comparison.md'

def collect_results(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob('*.json')):
        if 'report' in path.name.lower() or 'result' in path.name.lower():
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                row = {'file': str(path), **{key: value for key, value in payload.items() if isinstance(value, (str, int, float, bool, type(None)))}}
                rows.append(row)
    return rows

def _md(rows: list[dict[str, Any]]) -> str:
    lines = ['# AutoRAG retrieval comparison', '', f'Collected files: **{len(rows)}**', '']
    if rows:
        headers = sorted({key for row in rows for key in row.keys()})
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('|' + '|'.join(('---' for _ in headers)) + '|')
        for row in rows:
            lines.append('| ' + ' | '.join((str(row.get(key, '')) for key in headers)) + ' |')
    return '\n'.join(lines) + '\n'

def main() -> None:
    parser = argparse.ArgumentParser(description='Collect AutoRAG result JSON files into comparison reports.')
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR)
    parser.add_argument('--out-csv', default=DEFAULT_OUT_CSV)
    parser.add_argument('--out-json', default=DEFAULT_OUT_JSON)
    parser.add_argument('--out-md', default=DEFAULT_OUT_MD)
    args = parser.parse_args()
    root = project_root()
    rows = collect_results(root / args.input_dir)
    Path(root / args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / args.out_csv, index=False)
    write_json(root / args.out_json, {'rows': rows})
    write_text(root / args.out_md, _md(rows))
    logger.info({'rows': len(rows)})
if __name__ == '__main__':
    main()
