from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd
_CURRENT = Path(__file__).resolve()
ROOT = next((parent for parent in _CURRENT.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), Path.cwd())
from scripts.rag_corpus.common.io import ensure_dir, write_json, write_text
DEFAULT_PROJECT_DIR = 'rag_corpus/autorag/trials'
DEFAULT_OUTPUT_DIR = 'rag_corpus/autorag/report'

def _read_summary(path: Path, project_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rel = path.relative_to(project_dir) if path.is_relative_to(project_dir) else path.name
    frame.insert(0, 'autorag_summary_file', str(rel))
    return frame

def collect_autorag_results(*, project_dir: Path, output_dir: Path) -> dict[str, Any]:
    summary_paths = sorted(project_dir.rglob('summary.csv'))
    if not summary_paths:
        raise FileNotFoundError(f'No AutoRAG summary.csv files found under {project_dir}. Run autorag evaluate first.')
    frames = [_read_summary(path, project_dir) for path in summary_paths]
    combined = pd.concat(frames, ignore_index=True)
    ensure_dir(output_dir)
    csv_path = output_dir / 'autorag_summary.csv'
    json_path = output_dir / 'autorag_summary.json'
    md_path = output_dir / 'autorag_report.md'
    combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
    rows: list[dict[str, Any]] = combined.where(pd.notnull(combined), None).to_dict(orient='records')
    report = {'project_dir': str(project_dir), 'summary_files': [str(path) for path in summary_paths], 'rows': rows, 'output_csv': str(csv_path)}
    write_json(json_path, report)
    write_text(md_path, _markdown(summary_paths=summary_paths, combined=combined, csv_path=csv_path))
    return report

def _markdown(*, summary_paths: list[Path], combined: pd.DataFrame, csv_path: Path) -> str:
    lines = ['# ViRAGE AutoRAG evaluation report', '', 'This report only aggregates AutoRAG `summary.csv` files. It does not recompute retrieval metrics.', '', f'Summary files: **{len(summary_paths)}**', f'Rows: **{len(combined)}**', f'Combined CSV: `{csv_path}`', '']
    preview = combined.head(20)
    if not preview.empty:
        headers = [str(col) for col in preview.columns[:12]]
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('|' + '|'.join(('---' for _ in headers)) + '|')
        for _, row in preview.iterrows():
            lines.append('| ' + ' | '.join((str(row.get(col, '')).replace('|', '/') for col in headers)) + ' |')
    return '\n'.join(lines) + '\n'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect AutoRAG summary.csv files into one report without recomputing metrics.')
    parser.add_argument('--project-dir', default=DEFAULT_PROJECT_DIR)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    report = collect_autorag_results(project_dir=ROOT / args.project_dir, output_dir=ROOT / args.output_dir)
    logger.info(json.dumps({'rows': len(report['rows']), 'output_csv': report['output_csv']}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
