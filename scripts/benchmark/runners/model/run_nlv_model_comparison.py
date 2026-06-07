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
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "ui" / "config" / "benchmark"
DEFAULT_CASES_PATH = PROJECT_ROOT / "external_datasets" / "nlv_corpus"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "model_nlv"


@dataclass(frozen=True)
class ConfigRun:
    config_path: Path
    config_slug: str
    output_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int
    stdout_path: Path
    stderr_path: Path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    runs = build_config_runs(configs=resolve_configs(args.configs, args.config_dir), output_root=args.output_root.resolve())
    validate_input_paths(cases_path=args.cases.resolve(), runs=runs)
    logger.info("NLV configs: %s", len(runs))
    logger.info("NLV cases: %s", args.cases.resolve())
    completed: list[dict[str, object]] = []
    for run in runs:
        logger.info("\n=== NLV config=%s ===", run.config_path.name)
        result = invoke_nlv_runner(
            python_executable=args.python_executable,
            config_path=run.config_path,
            cases_path=args.cases.resolve(),
            output_dir=run.output_dir,
            logs_dir=run.logs_dir,
            limit=args.limit,
            seed=args.seed,
            resume=args.resume,
            retry_failed=args.retry_failed,
            disable_vlm_loop=args.disable_vlm_loop,
            disable_analytics_tail=args.disable_analytics_tail,
            sampling=args.sampling,
            max_per_chart_type=args.max_per_chart_type,
        )
        record = write_run_status(run=run, result=result)
        completed.append(record)
        log_run_result(run=run, result=result)
        if result.exit_code != 0 and not args.continue_on_error:
            raise RuntimeError(f"NLV benchmark failed for config '{run.config_path.name}'.")
    write_launcher_report(output_root=args.output_root.resolve(), records=completed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLV benchmark for local_test TOML configurations.")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--configs", nargs="*", type=Path, default=None, help="Explicit TOML configs. Defaults to ui/config/benchmark/local_test_*.toml.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None, help="Optional random case limit. Without --limit all NLV cases are used.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=["random", "stratified_chart_type"], default="random")
    parser.add_argument("--max-per-chart-type", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--disable-vlm-loop", action="store_true")
    parser.add_argument("--disable-analytics-tail", action="store_true", default=True)
    return parser.parse_args()


def resolve_configs(configs: Sequence[Path] | None, config_dir: Path) -> list[Path]:
    if configs:
        return [path.resolve() for path in configs]
    found = sorted(config_dir.resolve().glob("local_test_*.toml"))
    if not found:
        raise FileNotFoundError(f"No local_test_*.toml configs found in {config_dir.resolve()}.")
    return found


def validate_input_paths(*, cases_path: Path, runs: Sequence[ConfigRun]) -> None:
    if not BENCHMARK_RUNNER.exists():
        raise FileNotFoundError(f"NLV runner not found: {BENCHMARK_RUNNER}")
    if not cases_path.exists():
        raise FileNotFoundError(f"NLV cases path not found: {cases_path}")
    for run in runs:
        if not run.config_path.exists():
            raise FileNotFoundError(f"Config not found: {run.config_path}")
        with run.config_path.open("rb") as file:
            tomllib.load(file)


def build_config_runs(*, configs: Sequence[Path], output_root: Path) -> list[ConfigRun]:
    return [
        ConfigRun(
            config_path=config,
            config_slug=config.stem,
            output_dir=output_root / config.stem,
            logs_dir=output_root / config.stem / "logs",
        )
        for config in configs
    ]


def invoke_nlv_runner(
    *,
    python_executable: str,
    config_path: Path,
    cases_path: Path,
    output_dir: Path,
    logs_dir: Path,
    limit: int | None,
    seed: int,
    resume: bool,
    retry_failed: bool,
    disable_vlm_loop: bool,
    disable_analytics_tail: bool,
    sampling: str,
    max_per_chart_type: int | None,
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
        seed=seed,
        resume=resume,
        retry_failed=retry_failed,
        disable_vlm_loop=disable_vlm_loop,
        disable_analytics_tail=disable_analytics_tail,
        sampling=sampling,
        max_per_chart_type=max_per_chart_type,
    )
    logger.info("Command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return ProcessResult(exit_code=completed.returncode, stdout_path=stdout_path, stderr_path=stderr_path)


def build_runner_command(
    *,
    python_executable: str,
    config_path: Path,
    cases_path: Path,
    output_dir: Path,
    limit: int | None,
    seed: int,
    resume: bool,
    retry_failed: bool,
    disable_vlm_loop: bool,
    disable_analytics_tail: bool,
    sampling: str,
    max_per_chart_type: int | None,
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
        command.extend(["--limit", str(limit), "--shuffle", "--seed", str(seed)])
    command.extend(["--sampling", sampling])
    if max_per_chart_type is not None:
        command.extend(["--max-per-chart-type", str(max_per_chart_type)])
    if resume:
        command.append("--resume")
    if retry_failed:
        command.append("--retry-failed")
    if disable_vlm_loop:
        command.append("--disable-vlm-loop")
    if disable_analytics_tail:
        command.append("--disable-analytics-tail")
    return command


def write_run_status(*, run: ConfigRun, result: ProcessResult) -> dict[str, object]:
    benchmark_report = run.output_dir / "benchmark_report.json"
    payload: dict[str, object] = {
        "config": run.config_path.name,
        "config_slug": run.config_slug,
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


def log_run_result(*, run: ConfigRun, result: ProcessResult) -> None:
    status = "OK" if result.exit_code == 0 else "FAILED"
    logger.info("%s: config=%s output=%s", status, run.config_path.name, run.output_dir.as_posix())
    logger.info("stdout: %s", result.stdout_path.as_posix())
    logger.info("stderr: %s", result.stderr_path.as_posix())
    if result.exit_code != 0:
        logger.error(last_text_lines(result.stderr_path, limit=30))


def last_text_lines(path: Path, *, limit: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def write_launcher_report(*, output_root: Path, records: Sequence[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "model_nlv_runs.json").write_text(json.dumps(list(records), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
