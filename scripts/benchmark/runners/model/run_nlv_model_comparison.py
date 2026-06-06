from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
PROJECT_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_RUNNER = PROJECT_ROOT / 'scripts' / 'benchmark' / 'runners' / 'chart' / 'run_vegachat_compatible_benchmark.py'
DEFAULT_BASE_CONFIG = PROJECT_ROOT / 'ui' / 'config' / 'benchmark' / 'project-gemma3_local-bench_rag.toml'
DEFAULT_CASES_PATH = PROJECT_ROOT / 'external_datasets' / 'nlv_corpus'
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'artifacts' / 'model_nlv'
DEFAULT_MODELS: tuple[str, ...] = ('qwen3.5:latest', 'qwen3.5:4b', 'gemma3:4b', 'qwen2.5vl:latest', 'gemma3:12b-it-q4_K_M')
MODEL_SECTIONS: tuple[str, ...] = ('reasoning_model', 'spec_model', 'vlm_model')

@dataclass(frozen=True)
class ModelRun:
    model: str
    model_slug: str
    config_path: Path
    output_dir: Path
    logs_dir: Path

@dataclass(frozen=True)
class ProcessResult:
    exit_code: int
    stdout_path: Path
    stderr_path: Path

class TomlModelOverrideError(ValueError):
    pass

def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    runs = build_model_runs(models=args.models, output_root=args.output_root.resolve())
    validate_input_paths(project_root=project_root, base_config=args.base_config.resolve(), cases_path=args.cases.resolve())
    completed: list[dict[str, object]] = []
    for run in runs:
        write_all_in_one_config(base_config=args.base_config.resolve(), output_config=run.config_path, model=run.model)
        logger.info(f'\n=== NLV model={run.model} ===')
        process_result = invoke_nlv_runner(python_executable=args.python_executable, config_path=run.config_path, cases_path=args.cases.resolve(), output_dir=run.output_dir, logs_dir=run.logs_dir, limit=args.limit, resume=args.resume, retry_failed=args.retry_failed, disable_vlm_loop=args.disable_vlm_loop, disable_analytics_tail=args.disable_analytics_tail)
        record = write_run_status(run=run, result=process_result)
        completed.append(record)
        print_run_result(run=run, result=process_result)
        if process_result.exit_code != 0 and (not args.continue_on_error):
            raise RuntimeError(f"NLV benchmark failed for model '{run.model}'.")
    write_launcher_report(output_root=args.output_root.resolve(), records=completed)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run full NLV benchmark for selected local all-in-one models.')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--base-config', type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument('--cases', type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    parser.add_argument('--python-executable', default=sys.executable)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--disable-vlm-loop', action='store_true')
    parser.add_argument('--disable-analytics-tail', action='store_true', default=True)
    return parser.parse_args()

def validate_input_paths(*, project_root: Path, base_config: Path, cases_path: Path) -> None:
    if not project_root.exists():
        raise FileNotFoundError(f'Project root not found: {project_root}')
    if not BENCHMARK_RUNNER.exists():
        raise FileNotFoundError(f'NLV runner not found: {BENCHMARK_RUNNER}')
    if not base_config.exists():
        raise FileNotFoundError(f'Base TOML config not found: {base_config}')
    if not cases_path.exists():
        raise FileNotFoundError(f'NLV cases path not found: {cases_path}')

def build_model_runs(*, models: Sequence[str], output_root: Path) -> list[ModelRun]:
    return [ModelRun(model=model, model_slug=slugify_model_name(model), config_path=output_root / 'configs' / f'local_all_in_one_{slugify_model_name(model)}.toml', output_dir=output_root / f'local_all_in_one_{slugify_model_name(model)}', logs_dir=output_root / f'local_all_in_one_{slugify_model_name(model)}' / 'logs') for model in models]

def slugify_model_name(model: str) -> str:
    replacements = {'/': '_', ':': '_', '.': '_', '-': '-'}
    value = model.strip()
    for source, replacement in replacements.items():
        value = value.replace(source, replacement)
    if not value:
        raise ValueError('Model name must be non-empty.')
    return value

def write_all_in_one_config(*, base_config: Path, output_config: Path, model: str) -> None:
    lines = base_config.read_text(encoding='utf-8-sig').splitlines()
    for section in MODEL_SECTIONS:
        lines = replace_model_in_section(lines=lines, section=section, model=model)
    text = '\n'.join(lines) + '\n'
    tomllib.loads(text)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(text, encoding='utf-8')

def replace_model_in_section(*, lines: list[str], section: str, model: str) -> list[str]:
    updated: list[str] = []
    inside_section = False
    model_was_updated = False
    for line in lines:
        section_name = parse_section_name(line)
        if section_name is not None:
            inside_section = section_name == section
            updated.append(line)
            continue
        if inside_section and is_model_line(line):
            updated.append(f'model = "{model}"')
            model_was_updated = True
            continue
        updated.append(line)
    if not model_was_updated:
        raise TomlModelOverrideError(f'Section [{section}] with model field was not found.')
    return updated

def parse_section_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith('[') or not stripped.endswith(']'):
        return None
    if stripped.startswith('[['):
        return None
    return stripped[1:-1].strip()

def is_model_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith('model') and stripped.partition('=')[0].strip() == 'model'

def invoke_nlv_runner(*, python_executable: str, config_path: Path, cases_path: Path, output_dir: Path, logs_dir: Path, limit: int | None, resume: bool, retry_failed: bool, disable_vlm_loop: bool, disable_analytics_tail: bool) -> ProcessResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / 'stdout.txt'
    stderr_path = logs_dir / 'stderr.txt'
    command = build_runner_command(python_executable=python_executable, config_path=config_path, cases_path=cases_path, output_dir=output_dir, limit=limit, resume=resume, retry_failed=retry_failed, disable_vlm_loop=disable_vlm_loop, disable_analytics_tail=disable_analytics_tail)
    started = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stdout_path.write_text(started.stdout, encoding='utf-8')
    stderr_path.write_text(started.stderr, encoding='utf-8')
    return ProcessResult(exit_code=started.returncode, stdout_path=stdout_path, stderr_path=stderr_path)

def build_runner_command(*, python_executable: str, config_path: Path, cases_path: Path, output_dir: Path, limit: int | None, resume: bool, retry_failed: bool, disable_vlm_loop: bool, disable_analytics_tail: bool) -> list[str]:
    command = [python_executable, str(BENCHMARK_RUNNER), '--cases', str(cases_path), '--config', str(config_path), '--output-dir', str(output_dir), '--nlv-mode', 'single_turn']
    if limit is not None:
        command.extend(['--limit', str(limit)])
    if resume:
        command.append('--resume')
    if retry_failed:
        command.append('--retry-failed')
    if disable_vlm_loop:
        command.append('--disable-vlm-loop')
    if disable_analytics_tail:
        command.append('--disable-analytics-tail')
    return command

def write_run_status(*, run: ModelRun, result: ProcessResult) -> dict[str, object]:
    benchmark_report = run.output_dir / 'benchmark_report.json'
    payload: dict[str, object] = {'model': run.model, 'model_slug': run.model_slug, 'exit_code': result.exit_code, 'output_dir': run.output_dir.as_posix(), 'config_path': run.config_path.as_posix(), 'stdout_path': result.stdout_path.as_posix(), 'stderr_path': result.stderr_path.as_posix(), 'benchmark_report': benchmark_report.as_posix(), 'benchmark_report_exists': benchmark_report.exists()}
    status_path = run.output_dir / 'model_run_status.json'
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload

def print_run_result(*, run: ModelRun, result: ProcessResult) -> None:
    status = 'OK' if result.exit_code == 0 else 'FAILED'
    logger.info(f'{status}: model={run.model} output={run.output_dir.as_posix()}')
    if result.exit_code != 0:
        logger.info(f'stderr: {result.stderr_path.as_posix()}')

def write_launcher_report(*, output_root: Path, records: Sequence[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'model_nlv_runs.json').write_text(json.dumps(list(records), ensure_ascii=False, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
