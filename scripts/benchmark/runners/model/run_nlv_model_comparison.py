from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_RUNNER = PROJECT_ROOT / "scripts" / "benchmark" / "runners" / "chart" / "run_vegachat_compatible_benchmark.py"
DEFAULT_CASES_PATH = PROJECT_ROOT / "external_datasets" / "nlv_corpus"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "model_nlv"
DEFAULT_LOCAL_TEST_CONFIGS: tuple[Path, ...] = (
    PROJECT_ROOT / "ui" / "config" / "benchmark" / "local_test_gemma3_4b_safe.toml",
    PROJECT_ROOT / "ui" / "config" / "benchmark" / "local_test_gemma3_4b_extended.toml",
    PROJECT_ROOT / "ui" / "config" / "benchmark" / "local_test_gemma4_e2b_qat.toml",
    PROJECT_ROOT / "ui" / "config" / "benchmark" / "local_test_gemma4_e4b_qat.toml",
    PROJECT_ROOT / "ui" / "config" / "benchmark" / "local_test_qwen2_5_coder_7b_safe.toml",
)
MODEL_SECTIONS: tuple[str, ...] = ("reasoning_model", "spec_model", "vlm_model")


@dataclass(frozen=True)
class ModelRun:
    label: str
    model_summary: str
    config_path: Path
    output_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int
    stdout_path: Path
    stderr_path: Path


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_paths = resolve_config_paths(args.configs, project_root=project_root)
    validate_input_paths(project_root=project_root, cases_path=args.cases.resolve(), config_paths=config_paths)
    runs = build_config_runs(config_paths=config_paths, output_root=args.output_root.resolve())
    completed = run_all_configs(args=args, project_root=project_root, runs=runs)
    write_launcher_report(output_root=args.output_root.resolve(), records=completed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full NLV benchmark for local test TOML configurations.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--configs", nargs="+", type=Path, default=list(DEFAULT_LOCAL_TEST_CONFIGS))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--disable-vlm-loop", action="store_true")
    parser.add_argument("--disable-analytics-tail", action="store_true", default=True)
    return parser.parse_args()


def resolve_config_paths(configs: Sequence[Path], *, project_root: Path) -> list[Path]:
    paths = [path if path.is_absolute() else project_root / path for path in configs]
    return [path.resolve() for path in paths]


def validate_input_paths(*, project_root: Path, cases_path: Path, config_paths: Sequence[Path]) -> None:
    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    if not BENCHMARK_RUNNER.exists():
        raise FileNotFoundError(f"NLV runner not found: {BENCHMARK_RUNNER}")
    if not cases_path.exists():
        raise FileNotFoundError(f"NLV cases path not found: {cases_path}")
    for config_path in config_paths:
        if not config_path.exists():
            raise FileNotFoundError(f"Local test TOML config not found: {config_path}")


def build_config_runs(*, config_paths: Sequence[Path], output_root: Path) -> list[ModelRun]:
    return [build_config_run(config_path=config_path, output_root=output_root) for config_path in config_paths]


def build_config_run(*, config_path: Path, output_root: Path) -> ModelRun:
    label = config_path.stem
    output_dir = output_root / label
    return ModelRun(
        label=label,
        model_summary=read_model_summary(config_path),
        config_path=config_path,
        output_dir=output_dir,
        logs_dir=output_dir / "logs",
    )


def read_model_summary(config_path: Path) -> str:
    payload = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))
    models = [read_section_model(payload, section) for section in MODEL_SECTIONS]
    unique_models = list(dict.fromkeys(model for model in models if model))
    return unique_models[0] if len(unique_models) == 1 else ", ".join(unique_models)


def read_section_model(payload: dict[str, object], section: str) -> str | None:
    value = payload.get(section)
    if not isinstance(value, dict):
        return None
    model = value.get("model")
    return model if isinstance(model, str) else None


def run_all_configs(*, args: argparse.Namespace, project_root: Path, runs: Sequence[ModelRun]) -> list[dict[str, object]]:
    completed: list[dict[str, object]] = []
    for run in runs:
        logger.info("\n=== NLV config=%s models=%s ===", run.label, run.model_summary)
        result = invoke_nlv_runner(
            python_executable=args.python_executable,
            project_root=project_root,
            config_path=run.config_path,
            cases_path=args.cases.resolve(),
            output_dir=run.output_dir,
            logs_dir=run.logs_dir,
            limit=args.limit,
            resume=args.resume,
            retry_failed=args.retry_failed,
            disable_vlm_loop=args.disable_vlm_loop,
            disable_analytics_tail=args.disable_analytics_tail,
        )
        completed.append(write_run_status(run=run, result=result))
        print_run_result(run=run, result=result)
        if result.exit_code != 0 and not args.continue_on_error:
            raise RuntimeError(f"NLV benchmark failed for config '{run.label}'.")
    return completed


def invoke_nlv_runner(
    *,
    python_executable: str,
    project_root: Path,
    config_path: Path,
    cases_path: Path,
    output_dir: Path,
    logs_dir: Path,
    limit: int | None,
    resume: bool,
    retry_failed: bool,
    disable_vlm_loop: bool,
    disable_analytics_tail: bool,
) -> ProcessResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "stdout.txt"
    stderr_path = logs_dir / "stderr.txt"
    command = build_runner_command(
        python_executable=python_executable,
        config_path=config_path,
        cases_path=cases_path,
        output_dir=output_dir,
        limit=limit,
        resume=resume,
        retry_failed=retry_failed,
        disable_vlm_loop=disable_vlm_loop,
        disable_analytics_tail=disable_analytics_tail,
    )
    process = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    return ProcessResult(exit_code=process.returncode, stdout_path=stdout_path, stderr_path=stderr_path)


def build_runner_command(
    *,
    python_executable: str,
    config_path: Path,
    cases_path: Path,
    output_dir: Path,
    limit: int | None,
    resume: bool,
    retry_failed: bool,
    disable_vlm_loop: bool,
    disable_analytics_tail: bool,
) -> list[str]:
    command = [
        python_executable,
        str(BENCHMARK_RUNNER),
        "--cases",
        str(cases_path),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--nlv-mode",
        "single_turn",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    append_boolean_runner_flags(
        command,
        resume=resume,
        retry_failed=retry_failed,
        disable_vlm_loop=disable_vlm_loop,
        disable_analytics_tail=disable_analytics_tail,
    )
    return command


def append_boolean_runner_flags(
    command: list[str],
    *,
    resume: bool,
    retry_failed: bool,
    disable_vlm_loop: bool,
    disable_analytics_tail: bool,
) -> None:
    if resume:
        command.append("--resume")
    if retry_failed:
        command.append("--retry-failed")
    if disable_vlm_loop:
        command.append("--disable-vlm-loop")
    if disable_analytics_tail:
        command.append("--disable-analytics-tail")


def write_run_status(*, run: ModelRun, result: ProcessResult) -> dict[str, object]:
    benchmark_report = run.output_dir / "benchmark_report.json"
    payload: dict[str, object] = {
        "config": run.label,
        "model_summary": run.model_summary,
        "exit_code": result.exit_code,
        "output_dir": run.output_dir.as_posix(),
        "config_path": run.config_path.as_posix(),
        "stdout_path": result.stdout_path.as_posix(),
        "stderr_path": result.stderr_path.as_posix(),
        "benchmark_report": benchmark_report.as_posix(),
        "benchmark_report_exists": benchmark_report.exists(),
    }
    status_path = run.output_dir / "model_run_status.json"
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def print_run_result(*, run: ModelRun, result: ProcessResult) -> None:
    status = "OK" if result.exit_code == 0 else "FAILED"
    logger.info("%s: config=%s output=%s", status, run.label, run.output_dir.as_posix())
    if result.exit_code != 0:
        logger.info("stderr: %s", result.stderr_path.as_posix())


def write_launcher_report(*, output_root: Path, records: Sequence[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "model_nlv_runs.json"
    report_path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
