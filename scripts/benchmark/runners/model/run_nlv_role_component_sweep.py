from __future__ import annotations

import argparse
import itertools
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
DEFAULT_TEMPLATE_CONFIG = PROJECT_ROOT / "ui" / "config" / "benchmark" / "local_test_gemma3_4b_safe.toml"
DEFAULT_CASES_PATH = PROJECT_ROOT / "external_datasets" / "nlv_corpus"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "model_nlv_role_sweep"
DEFAULT_REASONING_MODELS = ("gemma3:4b", "gemma4:e2b-it-qat", "gemma4:e4b-it-qat", "qwen2.5-coder:7b")
DEFAULT_SPEC_MODELS = ("gemma3:4b", "gemma4:e2b-it-qat", "gemma4:e4b-it-qat", "qwen2.5-coder:7b")
DEFAULT_VLM_MODELS = ("gemma3:4b", "gemma4:e2b-it-qat", "gemma4:e4b-it-qat")


@dataclass(frozen=True)
class RoleRun:
    reasoning_model: str
    spec_model: str
    vlm_model: str
    slug: str
    config_path: Path
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
    runs = build_role_runs(
        reasoning_models=args.reasoning_models,
        spec_models=args.spec_models,
        vlm_models=args.vlm_models,
        output_root=args.output_root.resolve(),
    )
    validate_input_paths(template_config=args.template_config.resolve(), cases_path=args.cases.resolve())
    logger.info("Role combinations: %s", len(runs))
    logger.info("NLV sample: limit=%s seed=%s", args.limit, args.seed)
    completed: list[dict[str, object]] = []
    for run in runs:
        logger.info("\n=== role sweep %s ===", run.slug)
        write_role_config(template_config=args.template_config.resolve(), output_config=run.config_path, run=run)
        result = invoke_runner(
            python_executable=args.python_executable,
            config_path=run.config_path,
            cases_path=args.cases.resolve(),
            output_dir=run.output_dir,
            logs_dir=run.logs_dir,
            limit=args.limit,
            seed=args.seed,
            resume=args.resume,
            retry_failed=args.retry_failed,
            disable_analytics_tail=args.disable_analytics_tail,
        )
        record = write_run_status(run=run, result=result)
        completed.append(record)
        log_run_result(run=run, result=result)
        if result.exit_code != 0 and not args.continue_on_error:
            raise RuntimeError(f"NLV role sweep failed for '{run.slug}'.")
    write_launcher_report(output_root=args.output_root.resolve(), records=completed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NLV role-component sweep on local models.")
    parser.add_argument("--template-config", type=Path, default=DEFAULT_TEMPLATE_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reasoning-models", nargs="+", default=list(DEFAULT_REASONING_MODELS))
    parser.add_argument("--spec-models", nargs="+", default=list(DEFAULT_SPEC_MODELS))
    parser.add_argument("--vlm-models", nargs="+", default=list(DEFAULT_VLM_MODELS))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--disable-analytics-tail", action="store_true", default=True)
    return parser.parse_args()


def validate_input_paths(*, template_config: Path, cases_path: Path) -> None:
    if not BENCHMARK_RUNNER.exists():
        raise FileNotFoundError(f"NLV runner not found: {BENCHMARK_RUNNER}")
    if not template_config.exists():
        raise FileNotFoundError(f"Template TOML config not found: {template_config}")
    if not cases_path.exists():
        raise FileNotFoundError(f"NLV cases path not found: {cases_path}")
    with template_config.open("rb") as file:
        tomllib.load(file)


def build_role_runs(
    *,
    reasoning_models: Sequence[str],
    spec_models: Sequence[str],
    vlm_models: Sequence[str],
    output_root: Path,
) -> list[RoleRun]:
    runs: list[RoleRun] = []
    for reasoning_model, spec_model, vlm_model in itertools.product(reasoning_models, spec_models, vlm_models):
        slug = "reasoning-{0}__spec-{1}__vlm-{2}".format(
            slugify(reasoning_model),
            slugify(spec_model),
            slugify(vlm_model),
        )
        runs.append(RoleRun(
            reasoning_model=reasoning_model,
            spec_model=spec_model,
            vlm_model=vlm_model,
            slug=slug,
            config_path=output_root / "configs" / f"{slug}.toml",
            output_dir=output_root / slug,
            logs_dir=output_root / slug / "logs",
        ))
    return runs


def write_role_config(*, template_config: Path, output_config: Path, run: RoleRun) -> None:
    lines = template_config.read_text(encoding="utf-8-sig").splitlines()
    lines = replace_section_value(lines, section="reasoning_model", key="model", value=run.reasoning_model)
    lines = replace_section_value(lines, section="spec_model", key="model", value=run.spec_model)
    lines = replace_section_value(lines, section="vlm_model", key="model", value=run.vlm_model)
    lines = replace_section_value(lines, section="settings", key="spec_generation_max_attempts", value="20", quote=False)
    lines = replace_section_value(lines, section="settings", key="spec_generation_response_parse_retries", value="20", quote=False)
    lines = replace_section_value(lines, section="settings", key="semantic_feedback_loop_enabled", value="true", quote=False)
    lines = replace_section_value(lines, section="settings", key="semantic_feedback_max_attempts", value="20", quote=False)
    lines = replace_section_value(lines, section="settings", key="analytics_tail_enabled", value="false", quote=False)
    lines = replace_section_value(lines, section="settings", key="enable_evaluation_summary", value="false", quote=False)
    text = "\n".join(lines) + "\n"
    tomllib.loads(text)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(text, encoding="utf-8")


def replace_section_value(lines: list[str], *, section: str, key: str, value: str, quote: bool = True) -> list[str]:
    rendered_value = json.dumps(value, ensure_ascii=False) if quote else value
    updated: list[str] = []
    inside = False
    replaced = False
    for line in lines:
        section_name = parse_section_name(line)
        if section_name is not None:
            if inside and not replaced:
                updated.append(f"{key} = {rendered_value}")
                replaced = True
            inside = section_name == section
            updated.append(line)
            continue
        if inside and line.strip().partition("=")[0].strip() == key:
            updated.append(f"{key} = {rendered_value}")
            replaced = True
            continue
        updated.append(line)
    if inside and not replaced:
        updated.append(f"{key} = {rendered_value}")
    return updated


def parse_section_name(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]") and not stripped.startswith("[["):
        return stripped[1:-1].strip()
    return None


def invoke_runner(
    *,
    python_executable: str,
    config_path: Path,
    cases_path: Path,
    output_dir: Path,
    logs_dir: Path,
    limit: int,
    seed: int,
    resume: bool,
    retry_failed: bool,
    disable_analytics_tail: bool,
) -> ProcessResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "stdout.txt"
    stderr_path = logs_dir / "stderr.txt"
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
        "--limit",
        str(limit),
        "--shuffle",
        "--seed",
        str(seed),
    ]
    if resume:
        command.append("--resume")
    if retry_failed:
        command.append("--retry-failed")
    if disable_analytics_tail:
        command.append("--disable-analytics-tail")
    logger.info("Command: %s", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return ProcessResult(exit_code=completed.returncode, stdout_path=stdout_path, stderr_path=stderr_path)


def write_run_status(*, run: RoleRun, result: ProcessResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "slug": run.slug,
        "reasoning_model": run.reasoning_model,
        "spec_model": run.spec_model,
        "vlm_model": run.vlm_model,
        "exit_code": result.exit_code,
        "output_dir": run.output_dir.as_posix(),
        "config_path": run.config_path.as_posix(),
        "stdout_path": result.stdout_path.as_posix(),
        "stderr_path": result.stderr_path.as_posix(),
        "benchmark_report": (run.output_dir / "benchmark_report.json").as_posix(),
        "benchmark_report_exists": (run.output_dir / "benchmark_report.json").exists(),
    }
    (run.output_dir / "role_sweep_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def log_run_result(*, run: RoleRun, result: ProcessResult) -> None:
    status = "OK" if result.exit_code == 0 else "FAILED"
    logger.info("%s: %s", status, run.slug)
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
    (output_root / "role_sweep_runs.json").write_text(json.dumps(list(records), ensure_ascii=False, indent=2), encoding="utf-8")


def slugify(model: str) -> str:
    value = model.strip().replace("/", "_").replace(":", "_").replace(".", "_")
    if not value:
        raise ValueError("Model name must be non-empty.")
    return value


if __name__ == "__main__":
    main()
