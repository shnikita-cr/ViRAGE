from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
DEFAULT_LOCAL_ALL_IN_ONE_MODELS: tuple[str, ...] = ('qwen3.5:latest', 'qwen3.5:4b', 'gemma3:4b', 'qwen2.5vl:3b', 'gemma3:12b-it-q4_K_M')
MODEL_SECTIONS: tuple[str, ...] = ('reasoning_model', 'spec_model', 'vlm_model')

@dataclass(frozen=True)
class SuiteDefinition:
    name: str
    default_execute: bool
    case_count: int

@dataclass(frozen=True)
class RunResult:
    model: str
    suite: str
    run_id: str
    exit_code: int
    stdout_path: Path
    stderr_path: Path

    @property
    def status(self) -> str:
        return 'completed' if self.exit_code == 0 else 'failed'

class BenchmarkSuiteError(RuntimeError):
    pass

class TomlConfigError(RuntimeError):
    pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run all available benchmark suites for local all-in-one models.')
    parser.add_argument('--runner-module', default='scripts.benchmark.runners.chart.run_virage_e2e_test_cases')
    parser.add_argument('--base-config', default='ui/config/benchmark/project-gemma3_local-bench_rag.toml')
    parser.add_argument('--suites-file', default='benchmarks/benchmark_suites.json')
    parser.add_argument('--cases-dir', default='benchmarks/cases')
    parser.add_argument('--output-root', default='artifacts/model_suites')
    parser.add_argument('--model', action='append', default=None)
    parser.add_argument('--suite', action='append', default=None)
    parser.add_argument('--include-disabled-suites', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()

def load_default_suites(suites_file: Path, cases_dir: Path, include_disabled: bool) -> list[SuiteDefinition]:
    manifest = load_suites_manifest(suites_file)
    case_counts = load_case_counts(cases_dir)
    suite_names = sorted(case_counts)
    suites = [build_suite_definition(name, manifest, case_counts) for name in suite_names]
    if include_disabled:
        return suites
    return [suite for suite in suites if suite.default_execute]

def load_suites_manifest(suites_file: Path) -> Mapping[str, Mapping[str, Any]]:
    if not suites_file.exists():
        raise FileNotFoundError(f'Benchmark suites file not found: {suites_file}')
    with suites_file.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BenchmarkSuiteError(f'Benchmark suites manifest must be a JSON object: {suites_file}')
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}

def load_case_counts(cases_dir: Path) -> dict[str, int]:
    if not cases_dir.exists():
        raise FileNotFoundError(f'Benchmark cases directory not found: {cases_dir}')
    counts: dict[str, int] = {}
    for path in sorted(cases_dir.glob('*.jsonl')):
        for row in read_jsonl(path):
            suite = str(row.get('suite') or path.stem)
            counts[suite] = counts.get(suite, 0) + 1
    return counts

def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield parse_jsonl_row(line, path, line_number)

def parse_jsonl_row(line: str, path: Path, line_number: int) -> Mapping[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise BenchmarkSuiteError(f'Invalid JSONL row in {path}:{line_number}: {error.msg}') from error
    if not isinstance(payload, dict):
        raise BenchmarkSuiteError(f'JSONL row must be an object in {path}:{line_number}')
    return payload

def build_suite_definition(suite_name: str, manifest: Mapping[str, Mapping[str, Any]], case_counts: Mapping[str, int]) -> SuiteDefinition:
    manifest_entry = manifest.get(suite_name, {})
    default_execute = bool(manifest_entry.get('default_execute', True))
    return SuiteDefinition(name=suite_name, default_execute=default_execute, case_count=int(case_counts.get(suite_name, 0)))

def select_suites(args: argparse.Namespace) -> list[SuiteDefinition]:
    suites_file = Path(args.suites_file)
    cases_dir = Path(args.cases_dir)
    case_counts = load_case_counts(cases_dir)
    manifest = load_suites_manifest(suites_file)
    if args.suite:
        return [build_suite_definition(name, manifest, case_counts) for name in args.suite]
    return load_default_suites(suites_file, cases_dir, bool(args.include_disabled_suites))

def validate_suites(suites: Sequence[SuiteDefinition]) -> None:
    if not suites:
        raise BenchmarkSuiteError('No benchmark suites selected.')
    empty_suites = [suite.name for suite in suites if suite.case_count <= 0]
    if empty_suites:
        raise BenchmarkSuiteError(f"Selected suites have no cases: {', '.join(empty_suites)}")

def selected_models(args: argparse.Namespace) -> tuple[str, ...]:
    if args.model:
        return tuple(args.model)
    return DEFAULT_LOCAL_ALL_IN_ONE_MODELS

def model_to_path_name(model: str) -> str:
    return model.replace('/', '_').replace(':', '_').replace('.', '_')

def create_all_in_one_config(base_config: Path, output_config: Path, model: str) -> None:
    if not base_config.exists():
        raise FileNotFoundError(f'Base config not found: {base_config}')
    lines = base_config.read_text(encoding='utf-8-sig').splitlines()
    if not lines:
        raise TomlConfigError(f'Base config is empty: {base_config}')
    for section_name in MODEL_SECTIONS:
        lines = set_toml_model(lines, section_name, model)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def set_toml_model(lines: Sequence[str], section_name: str, model: str) -> list[str]:
    inside_section = False
    updated = False
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            inside_section = stripped == f'[{section_name}]'
            output.append(line)
            continue
        if inside_section and stripped.startswith('model') and ('=' in stripped):
            output.append(f'model = "{model}"')
            updated = True
            continue
        output.append(line)
    if not updated:
        raise TomlConfigError(f'TOML section [{section_name}] with model field not found')
    return output

def run_benchmark(runner_module: str, config_path: Path, suite: SuiteDefinition, run_id: str, log_dir: Path) -> RunResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / 'stdout.txt'
    stderr_path = log_dir / 'stderr.txt'
    command = [sys.executable, '-m', runner_module, '--config', config_path.as_posix(), '--suite', suite.name, '--run-id', run_id, '--execute']
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    stdout_path.write_text(completed.stdout, encoding='utf-8')
    stderr_path.write_text(completed.stderr, encoding='utf-8')
    return RunResult(model=extract_model_from_run_id(run_id), suite=suite.name, run_id=run_id, exit_code=int(completed.returncode), stdout_path=stdout_path, stderr_path=stderr_path)

def extract_model_from_run_id(run_id: str) -> str:
    parts = run_id.split('/')
    if len(parts) < 2:
        return 'unknown'
    return parts[1].removeprefix('local_all_in_one_')

def print_plan(models: Sequence[str], suites: Sequence[SuiteDefinition]) -> None:
    logger.info('Selected local all-in-one models:')
    for model in models:
        logger.info(f'  - {model}')
    logger.info('Selected suites:')
    for suite in suites:
        logger.info(f'  - {suite.name}: {suite.case_count} cases')

def print_result(result: RunResult) -> None:
    logger.info(f'{result.status.upper()}: model={result.model} suite={result.suite} exit_code={result.exit_code}')
    logger.info(f'  stdout: {result.stdout_path.as_posix()}')
    logger.info(f'  stderr: {result.stderr_path.as_posix()}')

def write_summary(results: Sequence[RunResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{'model': result.model, 'suite': result.suite, 'run_id': result.run_id, 'status': result.status, 'exit_code': result.exit_code, 'stdout_path': result.stdout_path.as_posix(), 'stderr_path': result.stderr_path.as_posix()} for result in results]
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    write_summary_csv(rows, output_path.with_suffix('.csv'))

def write_summary_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text('', encoding='utf-8')
        return
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main() -> int:
    args = parse_args()
    runner_module = str(args.runner_module)
    base_config = Path(args.base_config)
    output_root = Path(args.output_root)
    models = selected_models(args)
    suites = select_suites(args)
    validate_suites(suites)
    print_plan(models, suites)
    if args.dry_run:
        return 0
    results: list[RunResult] = []
    failed = False
    for model in models:
        model_path_name = model_to_path_name(model)
        config_path = output_root / 'configs' / f'local_all_in_one_{model_path_name}.toml'
        create_all_in_one_config(base_config, config_path, model)
        for suite in suites:
            run_id = f'model_suites/local_all_in_one_{model_path_name}/{suite.name}'
            log_dir = output_root / f'local_all_in_one_{model_path_name}' / suite.name / 'logs'
            logger.info(f'\n=== model={model} | suite={suite.name} | cases={suite.case_count} ===')
            result = run_benchmark(runner_module, config_path, suite, run_id, log_dir)
            results.append(result)
            print_result(result)
            if result.exit_code != 0:
                failed = True
                if args.fail_fast:
                    write_summary(results, output_root / 'local_all_in_one_summary.json')
                    return result.exit_code
    write_summary(results, output_root / 'local_all_in_one_summary.json')
    return 1 if failed else 0
if __name__ == '__main__':
    raise SystemExit(main())
