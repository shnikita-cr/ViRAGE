from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
from pathlib import Path

from scripts.common.json_io import load_user_context as _load_user_context, write_json
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[1]
from src.application.config.project_config import load_project_config
from src.infrastructure.runtime import RuntimeContext
from src.llm.factory import build_chat_model
from src.orchestrator.analysis_planner import AnalysisPlanner
from src.orchestrator.final_summary_builder import FinalSummaryBuilder
from src.orchestrator.image_folder_preprocessor import ImageFolderPreprocessor
from src.orchestrator.contracts.models import OrchestratorReport
from src.orchestrator.subtask_runner import OrchestratorSubtaskRunner
from src.services.data.profile.data_profiler import DataProfilerService

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Plan and optionally execute up to three ViRAGE analytical subtasks for one dataset.')
    parser.add_argument('--config', required=True, help='Path to project TOML config.')
    parser.add_argument('--query', required=True, help='User analytical request.')
    parser.add_argument('--data-path', required=True, help='Path to input CSV/XLSX table or image folder.')
    parser.add_argument('--input-type', default='auto', choices=['auto', 'table', 'image_folder'], help='Input interpretation. auto treats directories as image folders and files as tables.')
    parser.add_argument('--run-id', default=None, help='Optional deterministic orchestrator run id.')
    parser.add_argument('--artifact-root', default=None, help='Override settings.artifact_root for this run.')
    parser.add_argument('--max-charts', type=int, default=3, choices=[1, 2, 3], help='Maximum analytical subtasks.')
    parser.add_argument('--execute', action='store_true', help='Execute generated subtasks with the ViRAGE pipeline.')
    parser.add_argument('--user-context-json', default=None, help='Optional JSON object with extra user context.')
    parser.add_argument('--user-context-file', default=None, help='Optional path to JSON object with extra user context.')
    return parser

def _resolve_input_type(value: str, data_path: str) -> str:
    if value != 'auto':
        return value
    path = Path(data_path)
    return 'image_folder' if path.is_dir() else 'table'

def _prepare_input_data(*, input_type: str, data_path: str, run_dir: Path) -> tuple[str, dict[str, Any]]:
    if input_type == 'table':
        return (data_path, {})
    if input_type != 'image_folder':
        raise ValueError(f'Unsupported input type: {input_type}')
    result = ImageFolderPreprocessor().process(input_dir=data_path, output_dir=run_dir / 'input' / 'image_folder')
    return (result.output_csv_path, {'original_input_path': result.input_path, 'preprocessed_data_path': result.output_csv_path, 'failed_images_path': result.failed_csv_path, 'preprocessing_report_path': result.report_path, 'found_images': result.found_images, 'processed_images': result.processed_images, 'failed_images': result.failed_images, 'feature_columns': result.feature_columns})

def _orchestrator_run_id(value: str | None) -> str:
    if value:
        return value
    from datetime import datetime
    from uuid import uuid4
    return datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '_orchestrator_' + uuid4().hex

def main(argv: list[str] | None=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_project_config(args.config)
    if args.artifact_root:
        config.settings.artifact_root = Path(args.artifact_root)
    run_id = _orchestrator_run_id(args.run_id)
    run_dir = config.settings.artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_type = _resolve_input_type(args.input_type, args.data_path)
    effective_data_path, preprocessing_payload = _prepare_input_data(input_type=input_type, data_path=args.data_path, run_dir=run_dir)
    request_payload = {'run_id': run_id, 'query': args.query, 'input_type': input_type, 'original_data_path': args.data_path, 'data_path': effective_data_path, 'max_charts': args.max_charts, 'execute': bool(args.execute), 'preprocessing': preprocessing_payload, 'user_context': _load_user_context(args.user_context_json, args.user_context_file)}
    write_json(run_dir / 'orchestrator_request.json', request_payload)
    reasoning_llm = build_chat_model(config.reasoning_model)
    planning_runtime = RuntimeContext(settings=config.settings, reasoning_llm=reasoning_llm)
    planning_runtime.current_run_id = run_id
    data_profile = DataProfilerService().invoke(effective_data_path, planning_runtime)
    data_profile_path = run_dir / 'data_profile.json'
    write_json(data_profile_path, data_profile.model_dump())
    plan = AnalysisPlanner(max_charts=args.max_charts).plan(user_query=args.query, data_path=effective_data_path, data_profile=data_profile, runtime=planning_runtime, user_context=request_payload['user_context'], input_type=input_type, original_input_path=args.data_path if input_type != 'table' else None, preprocessing_report_path=preprocessing_payload.get('preprocessing_report_path'))
    plan_path = run_dir / 'chart_plan.json'
    write_json(plan_path, plan.model_dump())
    report = OrchestratorReport(run_id=run_id, user_query=args.query, data_path=effective_data_path, input_type=input_type, original_input_path=args.data_path if input_type != 'table' else None, preprocessing_report_path=preprocessing_payload.get('preprocessing_report_path'), plan_path=plan_path.as_posix(), data_profile_path=data_profile_path.as_posix(), executed=bool(args.execute))
    if args.execute:
        subruns = OrchestratorSubtaskRunner(config).run(parent_run_id=run_id, plan=plan)
        report.subruns = subruns
    final_summary = FinalSummaryBuilder().build(plan=plan, report=report)
    summary_path = run_dir / 'final_summary.md'
    summary_path.write_text(final_summary, encoding='utf-8')
    report.final_summary_path = summary_path.as_posix()
    report_path = run_dir / 'orchestrator_report.json'
    write_json(report_path, report.model_dump())
    logger.info(json.dumps({'run_id': run_id, 'status': 'completed', 'executed': bool(args.execute), 'input_type': input_type, 'run_dir': run_dir.as_posix(), 'chart_plan': plan_path.as_posix(), 'orchestrator_report': report_path.as_posix(), 'subtask_count': len(plan.subtasks)}, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
