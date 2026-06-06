from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
from pathlib import Path

from scripts.common.json_io import load_user_context as _load_user_context, write_json
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.config.project_config import load_project_config

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run one ViRAGE pipeline request and save reproducible artifacts.')
    parser.add_argument('--config', required=True, help='Path to project TOML config.')
    parser.add_argument('--query', required=True, help='User analytical request.')
    parser.add_argument('--data-path', required=True, help='Path to input CSV/XLSX table.')
    parser.add_argument('--run-id', default=None, help='Optional deterministic run id.')
    parser.add_argument('--artifact-root', default=None, help='Override settings.artifact_root for this run.')
    parser.add_argument('--user-context-json', default=None, help='Optional JSON object with extra user context.')
    parser.add_argument('--user-context-file', default=None, help='Optional path to JSON object with extra user context.')
    return parser

def main(argv: list[str] | None=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_project_config(args.config)
    if args.artifact_root:
        config.settings.artifact_root = Path(args.artifact_root)
    request = PipelineRequest(query=args.query, data_path=args.data_path, user_context=_load_user_context(args.user_context_json, args.user_context_file), **{'run_id': args.run_id} if args.run_id else {})
    pipeline = ViRAGEPipeline.from_project_config(config)
    result = pipeline.invoke(request)
    run_dir = pipeline.runtime.ensure_run_dir(request.run_id)
    summary = {'run_id': result.run_id, 'status': 'completed', 'run_dir': run_dir.as_posix(), 'run_report': (run_dir / 'run_report.json').as_posix()}
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
