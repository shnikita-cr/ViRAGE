from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.common.json_io import write_json
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from scripts.run_orchestrator import main as run_orchestrator_main
from src.application.config.project_config import load_project_config
from src.benchmark.core.progress import BenchmarkStatusBar
from src.benchmark.core.statistics import mean as _mean, mean_bool as _rate
DEFAULT_CASES_DIR = Path('benchmarks/cases')
DEFAULT_SUITES_PATH = Path('benchmarks/benchmark_suites.json')
_FLOAT_METRIC_FIELDS = ('spec_score', 'vision_score', 'plot_area_usage_score', 'axis_domain_score', 'layout_compactness_score', 'repeat_axis_label_score', 'publication_layout_score', 'chart_quality_score', 'chart_quality_critical_count', 'chart_quality_warning_count', 'duration_seconds')
_BOOL_METRIC_FIELDS = ('valid_spec', 'render_success', 'empty_chart')

@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    data_path: str
    query: str
    suite: str = 'manual'
    input_modality: str = 'table'
    query_specificity: str = 'unspecified'
    analysis_task: str = 'unspecified'
    chart_family: str = 'unspecified'
    output_target: str = 'unspecified'
    expected_charts: str = 'unspecified'
    evaluation_mode: str = 'e2e'
    known_risks: list[str] = field(default_factory=list)
    expected_checks: list[str] = field(default_factory=list)
    user_context: dict[str, Any] = field(default_factory=dict)
    focus: list[str] = field(default_factory=list)
    comparison_group_id: str | None = None
    input_type: str | None = None

    def __post_init__(self) -> None:
        normalized_input = self.input_type or self.input_modality
        object.__setattr__(self, 'input_modality', normalized_input)
        object.__setattr__(self, 'input_type', normalized_input)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'BenchmarkCase':
        required = ['case_id', 'data_path', 'query']
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f'Benchmark case is missing required fields {missing}: {payload}')
        input_modality = payload.get('input_modality', payload.get('input_type', 'table'))
        return cls(case_id=str(payload['case_id']), data_path=str(payload['data_path']), query=str(payload['query']), suite=str(payload.get('suite', 'manual')), input_modality=str(input_modality), query_specificity=str(payload.get('query_specificity', 'unspecified')), analysis_task=str(payload.get('analysis_task', 'unspecified')), chart_family=str(payload.get('chart_family', 'unspecified')), output_target=str(payload.get('output_target', 'unspecified')), expected_charts=str(payload.get('expected_charts', 'unspecified')), evaluation_mode=str(payload.get('evaluation_mode', 'e2e')), known_risks=[str(item) for item in payload.get('known_risks', [])], expected_checks=[str(item) for item in payload.get('expected_checks', [])], user_context=dict(payload.get('user_context') or {}), focus=[str(item) for item in payload.get('focus', [])], comparison_group_id=str(payload.get('comparison_group_id') or '').strip() or None)

    def model_dump(self) -> dict[str, Any]:
        return {'case_id': self.case_id, 'suite': self.suite, 'input_modality': self.input_modality, 'query_specificity': self.query_specificity, 'analysis_task': self.analysis_task, 'chart_family': self.chart_family, 'output_target': self.output_target, 'expected_charts': self.expected_charts, 'evaluation_mode': self.evaluation_mode, 'known_risks': self.known_risks, 'data_path': self.data_path, 'query': self.query, 'expected_checks': self.expected_checks, 'user_context': self.user_context, 'focus': self.focus, 'comparison_group_id': self.comparison_group_id}
VirageE2ETestCase = BenchmarkCase

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid JSONL in {path}:{line_no}: {exc}') from exc
        if not isinstance(payload, dict):
            raise ValueError(f'JSONL row must be object in {path}:{line_no}')
        rows.append(payload)
    return rows

def load_suites(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Benchmark suites file must be a JSON object: {path}')
    return payload

def discover_case_files(cases_path: Path, suite_names: set[str] | None) -> list[Path]:
    if not cases_path.exists():
        raise FileNotFoundError(f'Cases path does not exist: {cases_path}')
    if cases_path.is_file():
        files = [cases_path]
    else:
        files = sorted(cases_path.glob('*.jsonl'))
    if suite_names:
        files = [path for path in files if path.stem in suite_names]
    return files

def load_cases(cases_dir: Path, *, suites: set[str] | None=None, case_ids: set[str] | None=None) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for path in discover_case_files(cases_dir, suites):
        cases.extend((BenchmarkCase.from_dict(row) for row in read_jsonl(path)))
    if suites:
        cases = [case for case in cases if case.suite in suites]
    if case_ids:
        cases = [case for case in cases if case.case_id in case_ids]
    duplicate_ids = sorted({case.case_id for case in cases if [item.case_id for item in cases].count(case.case_id) > 1})
    if duplicate_ids:
        raise ValueError(f'Duplicate case ids: {duplicate_ids}')
    return cases

def _case_input_type(case: BenchmarkCase) -> str:
    if case.input_modality == 'image_folder':
        return 'image_folder'
    if case.input_modality == 'table':
        return 'table'
    raise ValueError(f'Unsupported input_modality for E2E orchestrator runner: {case.input_modality}')

def _resolve_case_data_path(case: BenchmarkCase, *, image_folder_override: str | None) -> str:
    if case.input_modality == 'image_folder' and image_folder_override:
        return image_folder_override
    if case.data_path == '{image_folder}' and image_folder_override:
        return image_folder_override
    return case.data_path

def resolve_case_data_path(case: BenchmarkCase, *, image_folder: str | None, project_root: Path) -> Path:
    resolved = Path(_resolve_case_data_path(case, image_folder_override=image_folder))
    if resolved.is_absolute():
        return resolved
    return project_root / resolved

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'artifact_read_error': 'missing', 'artifact_path': path.as_posix()}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        return {'artifact_read_error': f'invalid_json:{exc}', 'artifact_path': path.as_posix()}
    if not isinstance(payload, dict):
        return {'artifact_read_error': 'not_object', 'artifact_path': path.as_posix()}
    return payload

def _read_subrun_reports(orchestrator_report: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for item in orchestrator_report.get('subruns') or []:
        if not isinstance(item, dict):
            continue
        path = item.get('run_report_path')
        if not path:
            continue
        payload = _read_json(Path(str(path)))
        if payload:
            reports.append(payload)
    return reports

def _maybe_float(value: Any) -> float | None:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _maybe_bool(value: Any) -> bool | None:
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'true', '1', 'yes', 'y'}:
            return True
        if lowered in {'false', '0', 'no', 'n'}:
            return False
    return bool(value)

def _aggregate_subrun_metrics(reports: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {'subrun_count': len(reports)}
    for field in _FLOAT_METRIC_FIELDS:
        values = [value for report in reports if (value := _maybe_float(report.get(field))) is not None]
        metrics[f'mean_{field}'] = _mean(values)
    for field in _BOOL_METRIC_FIELDS:
        values = [value for report in reports if (value := _maybe_bool(report.get(field))) is not None]
        metrics[f'{field}_rate'] = _rate(values)
    if reports:
        metrics['artifact_read_error_count'] = sum((1 for report in reports if report.get('artifact_read_error')))
        metrics['successful_subrun_count'] = sum((1 for report in reports if report.get('status') == 'completed'))
        metrics['partial_subrun_count'] = sum((1 for report in reports if report.get('status') == 'partial'))
        metrics['error_subrun_count'] = sum((1 for report in reports if report.get('status') not in {'completed', 'partial'}))
        metrics['partial_success_rate'] = (metrics['successful_subrun_count'] + 0.5 * metrics['partial_subrun_count']) / len(reports)
    else:
        metrics['artifact_read_error_count'] = 0
        metrics['successful_subrun_count'] = 0
        metrics['partial_subrun_count'] = 0
        metrics['error_subrun_count'] = 0
        metrics['partial_success_rate'] = None
    return metrics

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def _format_float(value: Any) -> str:
    number = _maybe_float(value)
    return '' if number is None else f'{number:.4f}'

def _benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_suite: dict[str, dict[str, Any]] = {}
    for row in rows:
        suite = str(row.get('suite') or 'unknown')
        bucket = by_suite.setdefault(suite, {'cases': 0, 'completed': 0, 'partial': 0, 'errors': 0})
        bucket['cases'] += 1
        if row.get('status') == 'completed':
            bucket['completed'] += 1
        elif row.get('status') == 'partial':
            bucket['partial'] += 1
        else:
            bucket['errors'] += 1
    metric_means: dict[str, float | None] = {}
    for key in ('valid_spec_rate', 'render_success_rate', 'empty_chart_rate', 'mean_spec_score', 'mean_vision_score', 'mean_plot_area_usage_score', 'mean_axis_domain_score', 'mean_layout_compactness_score', 'mean_repeat_axis_label_score', 'mean_publication_layout_score', 'mean_chart_quality_score', 'partial_success_rate'):
        values = [value for row in rows if (value := _maybe_float(row.get(key))) is not None]
        metric_means[key] = _mean(values)
    return {'total_cases': len(rows), 'completed_cases': sum((1 for row in rows if row.get('status') == 'completed')), 'partial_cases': sum((1 for row in rows if row.get('status') == 'partial')), 'error_cases': sum((1 for row in rows if row.get('status') not in {'completed', 'partial'})), 'by_suite': by_suite, 'metric_means': metric_means}

def _write_report(path: Path, *, rows: list[dict[str, Any]], summary: dict[str, Any], execute: bool) -> None:
    lines = ['# ViRAGE E2E benchmark report', '', f'Cases: **{len(rows)}**', f'Execute mode: **{execute}**', '']
    lines.append('## Summary')
    lines.append('')
    lines.append(f"- Completed cases: {summary['completed_cases']}")
    lines.append(f"- Partial cases: {summary['partial_cases']}")
    lines.append(f"- Error cases: {summary['error_cases']}")
    lines.append('')
    lines.append('## Metrics')
    lines.append('')
    lines.append('| Metric | Mean |')
    lines.append('|---|---:|')
    for key, value in (summary.get('metric_means') or {}).items():
        lines.append(f'| {key} | {_format_float(value)} |')
    lines.append('')
    lines.append('## Per-case results')
    lines.append('')
    lines.append('| Case | Suite | Modality | Task | Target | Status | Subruns | Valid spec | Empty chart | Spec | Vision | Publication | Run dir |')
    lines.append('|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|')
    for row in rows:
        lines.append('| ' + ' | '.join([str(row.get('case_id', '')), str(row.get('suite', '')), str(row.get('input_modality', '')), str(row.get('analysis_task', '')), str(row.get('output_target', '')), str(row.get('status', '')), str(row.get('subrun_count', '')), _format_float(row.get('valid_spec_rate')), _format_float(row.get('empty_chart_rate')), _format_float(row.get('mean_spec_score')), _format_float(row.get('mean_vision_score')), _format_float(row.get('mean_publication_layout_score')), str(row.get('run_dir', ''))]) + ' |')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def _case_user_context(case: BenchmarkCase) -> dict[str, Any]:
    context = dict(case.user_context)
    context.update({'benchmark_case': {'case_id': case.case_id, 'suite': case.suite, 'input_modality': case.input_modality, 'query_specificity': case.query_specificity, 'analysis_task': case.analysis_task, 'chart_family': case.chart_family, 'output_target': case.output_target, 'expected_charts': case.expected_charts, 'known_risks': case.known_risks, 'expected_checks': case.expected_checks, 'comparison_group_id': case.comparison_group_id}})
    if case.comparison_group_id:
        context['comparison_group_id'] = case.comparison_group_id
    return context

def run_case(case: BenchmarkCase, *, config_path: str, parent_run_id: str, execute: bool, image_folder_override: str | None) -> dict[str, Any]:
    case_run_id = f'{parent_run_id}/cases/{case.case_id}'
    data_path = _resolve_case_data_path(case, image_folder_override=image_folder_override)
    args = ['--config', config_path, '--query', case.query, '--data-path', data_path, '--input-type', _case_input_type(case), '--run-id', case_run_id, '--user-context-json', json.dumps(_case_user_context(case), ensure_ascii=False)]
    if execute:
        args.append('--execute')
    started = time.perf_counter()
    status = 'completed'
    error_type = ''
    error = ''
    if execute:
        try:
            exit_code = run_orchestrator_main(args)
            if exit_code != 0:
                status = 'error'
                error_type = 'NonZeroExit'
                error = f'run_orchestrator exited with code {exit_code}'
        except (RuntimeError, ValueError, OSError) as exc:
            status = 'error'
            error_type = type(exc).__name__
            error = str(exc)
    duration = time.perf_counter() - started
    config = load_project_config(config_path)
    run_dir = config.settings.artifact_root / case_run_id
    orchestrator_report_path = run_dir / 'orchestrator_report.json'
    chart_plan_path = run_dir / 'chart_plan.json'
    orchestrator_report = _read_json(orchestrator_report_path)
    chart_plan = _read_json(chart_plan_path)
    subrun_reports = _read_subrun_reports(orchestrator_report)
    metrics = _aggregate_subrun_metrics(subrun_reports)
    result: dict[str, Any] = {**case.model_dump(), 'status': status, 'error_type': error_type, 'error': error, 'duration_seconds_total': duration, 'run_id': case_run_id, 'run_dir': run_dir.as_posix(), 'chart_plan_path': chart_plan_path.as_posix() if chart_plan_path.exists() else '', 'orchestrator_report_path': orchestrator_report_path.as_posix() if orchestrator_report_path.exists() else '', 'planned_subtasks': len(chart_plan.get('subtasks') or []), **metrics}
    if status == 'completed' and subrun_reports:
        subrun_statuses = {str(report.get('status') or '') for report in subrun_reports}
        if any((report.get('artifact_read_error') for report in subrun_reports)):
            result['status'] = 'error'
            result['error_type'] = 'ArtifactReadError'
            result['error'] = 'At least one subrun report could not be read.'
        elif any((item not in {'completed', 'partial'} for item in subrun_statuses)):
            result['status'] = 'error'
        elif 'partial' in subrun_statuses:
            result['status'] = 'partial'
    elif status == 'completed' and execute and (not subrun_reports):
        result['status'] = 'error'
        result['error_type'] = 'MissingSubrunReports'
        result['error'] = 'No subrun run_report.json files were found.'
    return result

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run classified ViRAGE E2E benchmark suites.')
    parser.add_argument('--config', required=True, help='Path to project TOML config.')
    parser.add_argument('--cases-dir', default=str(DEFAULT_CASES_DIR), help='Directory with benchmark case JSONL files.')
    parser.add_argument('--suites-file', default=str(DEFAULT_SUITES_PATH), help='Benchmark suites manifest JSON.')
    parser.add_argument('--suite', action='append', default=None, help='Suite name to run. Can be repeated.')
    parser.add_argument('--case-id', action='append', default=None, help='Case id to run. Can be repeated.')
    parser.add_argument('--run-id', required=True, help='Parent benchmark run id.')
    parser.add_argument('--execute', action='store_true', help='Execute orchestrator subruns, not only plan cases.')
    parser.add_argument('--max-cases', type=int, default=None, help='Limit number of selected cases.')
    parser.add_argument('--image-folder', default=None, help='Override image folder path for image_folder cases.')
    parser.add_argument('--list-suites', action='store_true', help='Print available suites and exit.')
    parser.add_argument('--no-progress', action='store_true', help='Disable console progress bar.')
    return parser

def main(argv: list[str] | None=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    suites_manifest = load_suites(Path(args.suites_file))
    if args.list_suites:
        logger.info(json.dumps(suites_manifest, ensure_ascii=False, indent=2))
        return 0
    selected_suites = set(args.suite or []) or None
    selected_case_ids = set(args.case_id or []) or None
    cases = load_cases(Path(args.cases_dir), suites=selected_suites, case_ids=selected_case_ids)
    if args.max_cases is not None:
        cases = cases[:max(0, args.max_cases)]
    if not cases:
        raise ValueError('No benchmark cases selected.')
    config = load_project_config(args.config)
    report_dir = config.settings.artifact_root / args.run_id / 'e2e_cases_report'
    report_dir.mkdir(parents=True, exist_ok=True)
    request = {'run_id': args.run_id, 'config': args.config, 'cases_dir': str(args.cases_dir), 'suites_file': str(args.suites_file), 'selected_suites': sorted(selected_suites or []), 'selected_case_ids': sorted(selected_case_ids or []), 'execute': bool(args.execute), 'image_folder_override': args.image_folder, 'case_count': len(cases)}
    write_json(report_dir / 'benchmark_request.json', request)
    progress = BenchmarkStatusBar(total=len(cases), title='ViRAGE E2E cases', enabled=not args.no_progress)
    rows: list[dict[str, Any]] = []
    ok = 0
    errors = 0
    for index, case in enumerate(cases, start=1):
        progress.update(index - 1, ok=ok, errors=errors, stage='case', label=case.case_id)
        row = run_case(case, config_path=args.config, parent_run_id=args.run_id, execute=bool(args.execute), image_folder_override=args.image_folder)
        rows.append(row)
        if row.get('status') == 'completed':
            ok += 1
        elif row.get('status') == 'partial':
            ok += 1
        else:
            errors += 1
        progress.update(index, ok=ok, errors=errors, stage='case', label=case.case_id)
    progress.close(label='completed')
    summary = _benchmark_summary(rows)
    write_json(report_dir / 'benchmark_summary.json', summary)
    _write_jsonl(report_dir / 'per_case_results.jsonl', rows)
    _write_csv(report_dir / 'per_case_results.csv', rows)
    _write_report(report_dir / 'benchmark_report.md', rows=rows, summary=summary, execute=bool(args.execute))
    logger.info(json.dumps({'run_id': args.run_id, 'status': 'completed', 'cases': len(rows), 'errors': errors, 'report_dir': report_dir.as_posix(), 'benchmark_report': (report_dir / 'benchmark_report.md').as_posix(), 'per_case_results': (report_dir / 'per_case_results.csv').as_posix()}, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
