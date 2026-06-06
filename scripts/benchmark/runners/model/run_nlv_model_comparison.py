from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.benchmark.datasets.datasets import load_benchmark_cases
from src.benchmark.core.models import BenchmarkCase

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_RUNNER = PROJECT_ROOT / "scripts" / "benchmark" / "runners" / "chart" / "run_vegachat_compatible_benchmark.py"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "ui" / "config" / "benchmark"
DEFAULT_CASES_PATH = PROJECT_ROOT / "external_datasets" / "nlv_corpus"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "model_nlv"
DEFAULT_CONFIG_GLOB = "local_test_*.toml"


@dataclass(frozen=True)
class ConfigRun:
    config_path: Path
    config_name: str
    output_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int
    stdout_path: Path
    stderr_path: Path


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    config_paths = discover_config_paths(args.configs, config_dir=args.config_dir.resolve(), pattern=args.config_glob)
    cases_path = prepare_cases_path(
        source=args.cases.resolve(),
        output_root=output_root,
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
        nlv_mode=args.nlv_mode,
    )
    validate_input_paths(config_paths=config_paths, cases_path=cases_path)
    runs = build_config_runs(config_paths=config_paths, output_root=output_root)
    completed: list[dict[str, object]] = []
    for run in runs:
        logger.info("=== NLV config=%s ===", run.config_name)
        process_result = invoke_nlv_runner(
            python_executable=args.python_executable,
            config_path=run.config_path,
            cases_path=cases_path,
            output_dir=run.output_dir,
            logs_dir=run.logs_dir,
            resume=args.resume,
            retry_failed=args.retry_failed,
            disable_vlm_loop=args.disable_vlm_loop,
            disable_analytics_tail=args.disable_analytics_tail,
            nlv_mode=args.nlv_mode,
        )
        record = write_run_status(run=run, result=process_result)
        completed.append(record)
        log_run_result(run=run, result=process_result)
        if process_result.exit_code != 0 and not args.continue_on_error:
            raise RuntimeError(f"NLV benchmark failed for config '{run.config_name}'.")
    write_launcher_report(output_root=output_root, records=completed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLV benchmark for local_test TOML configurations.")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--config-glob", default=DEFAULT_CONFIG_GLOB)
    parser.add_argument("--configs", nargs="*", type=Path, default=None)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--disable-vlm-loop", action="store_true")
    parser.add_argument("--disable-analytics-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nlv-mode", choices=["single_turn"], default="single_turn")
    return parser.parse_args()


def discover_config_paths(configs: Sequence[Path] | None, *, config_dir: Path, pattern: str) -> list[Path]:
    paths = [path.resolve() for path in configs] if configs else sorted(config_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No benchmark configs found in {config_dir} by pattern {pattern!r}.")
    return paths


def prepare_cases_path(
    *,
    source: Path,
    output_root: Path,
    limit: int | None,
    seed: int,
    shuffle: bool,
    nlv_mode: str,
) -> Path:
    if limit is None and not shuffle:
        return source
    if limit is None:
        return source
    if limit <= 0:
        raise ValueError("--limit must be positive.")
    cases = load_benchmark_cases(source, nlv_mode=nlv_mode)
    selected = sample_cases(cases, limit=limit, seed=seed, shuffle=shuffle)
    sampled_path = output_root / "sampled_cases" / f"nlv_{nlv_mode}_seed_{seed}_limit_{limit}.jsonl"
    write_cases_jsonl(sampled_path, selected)
    return sampled_path


def sample_cases(cases: Sequence[BenchmarkCase], *, limit: int, seed: int, shuffle: bool) -> list[BenchmarkCase]:
    if not cases:
        raise ValueError("No NLV cases were loaded.")
    ordered = list(cases)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ordered)
    return ordered[: min(limit, len(ordered))]


def write_cases_jsonl(path: Path, cases: Sequence[BenchmarkCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.model_dump(), ensure_ascii=False, default=str) + "\n")


def validate_input_paths(*, config_paths: Sequence[Path], cases_path: Path) -> None:
    if not BENCHMARK_RUNNER.exists():
        raise FileNotFoundError(f"NLV runner not found: {BENCHMARK_RUNNER}")
    if not cases_path.exists():
        raise FileNotFoundError(f"NLV cases path not found: {cases_path}")
    missing = [path for path in config_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing config files: " + ", ".join(path.as_posix() for path in missing))


def build_config_runs(*, config_paths: Sequence[Path], output_root: Path) -> list[ConfigRun]:
    return [
        ConfigRun(
            config_path=config_path,
            config_name=config_path.stem,
            output_dir=output_root / config_path.stem,
            logs_dir=output_root / config_path.stem / "logs",
        )
        for config_path in config_paths
    ]


def invoke_nlv_runner(
    *,
    python_executable: str,
    config_path: Path,
    cases_path: Path,
    output_dir: Path,
    logs_dir: Path,
    resume: bool,
    retry_failed: bool,
    disable_vlm_loop: bool,
    disable_analytics_tail: bool,
    nlv_mode: str,
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
        resume=resume,
        retry_failed=retry_failed,
        disable_vlm_loop=disable_vlm_loop,
        disable_analytics_tail=disable_analytics_tail,
        nlv_mode=nlv_mode,
    )
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    return ProcessResult(exit_code=result.returncode, stdout_path=stdout_path, stderr_path=stderr_path)


def build_runner_command(
    *,
    python_executable: str,
    config_path: Path,
    cases_path: Path,
    output_dir: Path,
    resume: bool,
    retry_failed: bool,
    disable_vlm_loop: bool,
    disable_analytics_tail: bool,
    nlv_mode: str,
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
        nlv_mode,
    ]
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
        "config_name": run.config_name,
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
    logger.info("%s: config=%s output=%s", status, run.config_name, run.output_dir.as_posix())
    if result.exit_code != 0:
        logger.info("stderr: %s", result.stderr_path.as_posix())


def write_launcher_report(*, output_root: Path, records: Sequence[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "model_nlv_runs.json").write_text(
        json.dumps(list(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
