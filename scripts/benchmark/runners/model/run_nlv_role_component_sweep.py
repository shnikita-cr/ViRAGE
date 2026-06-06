from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from scripts.benchmark.runners.model.run_nlv_model_comparison import (
    BENCHMARK_RUNNER,
    DEFAULT_CASES_PATH,
    PROJECT_ROOT,
    prepare_cases_path,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "model_nlv_role_sweep"
DEFAULT_GPU_RAM_GB = 8.0
DEFAULT_SAMPLE_LIMIT = 20
DEFAULT_SEED = 42
DEFAULT_REASONING_MODELS: tuple[str, ...] = (
    "gemma3:4b",
    "gemma4:e2b-it-qat",
    "gemma4:e4b-it-qat",
    "qwen2.5-coder:7b",
)
DEFAULT_SPEC_MODELS: tuple[str, ...] = DEFAULT_REASONING_MODELS
DEFAULT_VLM_MODELS: tuple[str, ...] = (
    "gemma3:4b",
    "gemma4:e2b-it-qat",
    "gemma4:e4b-it-qat",
)


@dataclass(frozen=True)
class RoleSweepRun:
    name: str
    reasoning_model: str
    spec_model: str
    vlm_model: str
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
    output_root = args.output_root.resolve()
    cases_path = prepare_cases_path(
        source=args.cases.resolve(),
        output_root=output_root,
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
        nlv_mode=args.nlv_mode,
    )
    validate_runner_inputs(cases_path)
    runs = build_role_sweep_runs(
        reasoning_models=args.reasoning_models,
        spec_models=args.spec_models,
        vlm_models=args.vlm_models,
        output_root=output_root,
    )
    completed: list[dict[str, object]] = []
    for run in runs:
        write_role_sweep_config(run, gpu_ram_gb=args.gpu_ram_gb)
        logger.info(
            "=== NLV role sweep=%s reasoning=%s spec=%s vlm=%s ===",
            run.name,
            run.reasoning_model,
            run.spec_model,
            run.vlm_model,
        )
        result = invoke_runner(
            python_executable=args.python_executable,
            run=run,
            cases_path=cases_path,
            resume=args.resume,
            retry_failed=args.retry_failed,
            disable_vlm_loop=args.disable_vlm_loop,
            disable_analytics_tail=args.disable_analytics_tail,
            nlv_mode=args.nlv_mode,
        )
        record = write_run_status(run=run, result=result)
        completed.append(record)
        log_run_result(run=run, result=result)
        if result.exit_code != 0 and not args.continue_on_error:
            raise RuntimeError(f"NLV role sweep failed for '{run.name}'.")
    write_launcher_report(output_root=output_root, records=completed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLV role-component sweep on random NLV examples.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--reasoning-models", nargs="+", default=list(DEFAULT_REASONING_MODELS))
    parser.add_argument("--spec-models", nargs="+", default=list(DEFAULT_SPEC_MODELS))
    parser.add_argument("--vlm-models", nargs="+", default=list(DEFAULT_VLM_MODELS))
    parser.add_argument("--limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-ram-gb", type=float, default=DEFAULT_GPU_RAM_GB)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--disable-vlm-loop", action="store_true")
    parser.add_argument("--disable-analytics-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nlv-mode", choices=["single_turn"], default="single_turn")
    return parser.parse_args()


def validate_runner_inputs(cases_path: Path) -> None:
    if not BENCHMARK_RUNNER.exists():
        raise FileNotFoundError(f"NLV runner not found: {BENCHMARK_RUNNER}")
    if not cases_path.exists():
        raise FileNotFoundError(f"NLV cases path not found: {cases_path}")


def build_role_sweep_runs(
    *,
    reasoning_models: Sequence[str],
    spec_models: Sequence[str],
    vlm_models: Sequence[str],
    output_root: Path,
) -> list[RoleSweepRun]:
    runs: list[RoleSweepRun] = []
    for reasoning_model, spec_model, vlm_model in itertools.product(reasoning_models, spec_models, vlm_models):
        name = "reasoning_{reasoning}__spec_{spec}__vlm_{vlm}".format(
            reasoning=slugify_model_name(reasoning_model),
            spec=slugify_model_name(spec_model),
            vlm=slugify_model_name(vlm_model),
        )
        runs.append(
            RoleSweepRun(
                name=name,
                reasoning_model=reasoning_model,
                spec_model=spec_model,
                vlm_model=vlm_model,
                config_path=output_root / "configs" / f"{name}.toml",
                output_dir=output_root / name,
                logs_dir=output_root / name / "logs",
            )
        )
    return runs


def slugify_model_name(model: str) -> str:
    cleaned = model.strip().replace("/", "_").replace(":", "_").replace(".", "_")
    if not cleaned:
        raise ValueError("Model name must be non-empty.")
    return cleaned


def write_role_sweep_config(run: RoleSweepRun, *, gpu_ram_gb: float) -> None:
    run.config_path.parent.mkdir(parents=True, exist_ok=True)
    run.config_path.write_text(build_config_text(run, gpu_ram_gb=gpu_ram_gb), encoding="utf-8")


def build_config_text(run: RoleSweepRun, *, gpu_ram_gb: float) -> str:
    return f'''mode = "benchmark"

[settings]
gpu_ram_gb = {gpu_ram_gb:.1f}
visrag_enabled = false
artifact_root = "./artifacts/model_nlv_role_sweep/{run.name}"
analytics_tail_enabled = false
enable_evaluation_summary = false
spec_generation_max_attempts = 20
spec_generation_response_parse_retries = 0
semantic_feedback_loop_enabled = true
semantic_feedback_max_attempts = 20
semantic_feedback_mode = "strict"
semantic_feedback_corpus_path = "./rag_corpus/feedback/visual_feedback_nlv_role_sweep.jsonl"

[reasoning_model]
provider = "ollama"
model = "{run.reasoning_model}"
base_url = "http://localhost:11434"
temperature = 0.0

[spec_model]
provider = "ollama"
model = "{run.spec_model}"
base_url = "http://localhost:11434"
temperature = 0.0

[vlm_model]
provider = "ollama"
model = "{run.vlm_model}"
base_url = "http://localhost:11434"
temperature = 0.0
'''


def invoke_runner(
    *,
    python_executable: str,
    run: RoleSweepRun,
    cases_path: Path,
    resume: bool,
    retry_failed: bool,
    disable_vlm_loop: bool,
    disable_analytics_tail: bool,
    nlv_mode: str,
) -> ProcessResult:
    run.logs_dir.mkdir(parents=True, exist_ok=True)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run.logs_dir / "stdout.txt"
    stderr_path = run.logs_dir / "stderr.txt"
    command = build_runner_command(
        python_executable=python_executable,
        run=run,
        cases_path=cases_path,
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
    run: RoleSweepRun,
    cases_path: Path,
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
        str(run.config_path),
        "--output-dir",
        str(run.output_dir),
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


def write_run_status(*, run: RoleSweepRun, result: ProcessResult) -> dict[str, object]:
    benchmark_report = run.output_dir / "benchmark_report.json"
    payload: dict[str, object] = {
        "name": run.name,
        "reasoning_model": run.reasoning_model,
        "spec_model": run.spec_model,
        "vlm_model": run.vlm_model,
        "exit_code": result.exit_code,
        "output_dir": run.output_dir.as_posix(),
        "config_path": run.config_path.as_posix(),
        "stdout_path": result.stdout_path.as_posix(),
        "stderr_path": result.stderr_path.as_posix(),
        "benchmark_report": benchmark_report.as_posix(),
        "benchmark_report_exists": benchmark_report.exists(),
    }
    (run.output_dir / "model_run_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def log_run_result(*, run: RoleSweepRun, result: ProcessResult) -> None:
    status = "OK" if result.exit_code == 0 else "FAILED"
    logger.info("%s: role_sweep=%s output=%s", status, run.name, run.output_dir.as_posix())
    if result.exit_code != 0:
        logger.info("stderr: %s", result.stderr_path.as_posix())


def write_launcher_report(*, output_root: Path, records: Sequence[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "role_component_sweep_runs.json").write_text(
        json.dumps(list(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
