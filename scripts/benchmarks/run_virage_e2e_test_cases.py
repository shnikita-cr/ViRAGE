from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.progress import ConsoleProgressBar

DEFAULT_CASES_PATH = PROJECT_ROOT / "benchmarks" / "virage_e2e_test_cases.jsonl"


@dataclass(slots=True)
class VirageE2ETestCase:
    case_id: str
    query: str
    data_path: str
    input_type: str = "table"
    focus: list[str] = field(default_factory=list)
    expected_checks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VirageE2ETestCase":
        case_id = str(payload.get("case_id", "")).strip()
        query = str(payload.get("query", "")).strip()
        data_path = str(payload.get("data_path", "")).strip()
        input_type = str(payload.get("input_type", "table")).strip() or "table"
        if not case_id:
            raise ValueError("Test case must contain non-empty case_id.")
        if not query:
            raise ValueError(f"Test case {case_id} must contain non-empty query.")
        if not data_path:
            raise ValueError(f"Test case {case_id} must contain non-empty data_path.")
        if input_type not in {"table", "image_folder", "auto"}:
            raise ValueError(f"Test case {case_id} has unsupported input_type={input_type!r}.")
        return cls(
            case_id=case_id,
            query=query,
            data_path=data_path,
            input_type=input_type,
            focus=[str(item) for item in payload.get("focus", [])],
            expected_checks=[str(item) for item in payload.get("expected_checks", [])],
            metadata={key: value for key, value in payload.items() if key not in {
                "case_id", "query", "data_path", "input_type", "focus", "expected_checks"
            }},
        )


@dataclass(slots=True)
class VirageE2ETestResult:
    case_id: str
    input_type: str
    data_path: str
    query: str
    status: str
    duration_seconds: float | None = None
    run_id: str | None = None
    run_dir: str | None = None
    chart_plan_path: str | None = None
    orchestrator_report_path: str | None = None
    subtask_count: int | None = None
    error: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    focus: list[str] = field(default_factory=list)
    expected_checks: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input_type": self.input_type,
            "data_path": self.data_path,
            "query": self.query,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "chart_plan_path": self.chart_plan_path,
            "orchestrator_report_path": self.orchestrator_report_path,
            "subtask_count": self.subtask_count,
            "error": self.error,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "focus": " | ".join(self.focus),
            "expected_checks": " | ".join(self.expected_checks),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed ViRAGE E2E/manual vulnerability test cases through the orchestrator."
    )
    parser.add_argument("--config", required=True, help="Path to project TOML config.")
    parser.add_argument("--cases", default=DEFAULT_CASES_PATH.as_posix(), help="Path to JSONL test cases.")
    parser.add_argument("--run-id", default=None, help="Optional deterministic benchmark run id.")
    parser.add_argument("--output-dir", default=None, help="Directory for benchmark summary artifacts.")
    parser.add_argument("--artifact-root", default=None, help="Artifact root passed to run_orchestrator.py.")
    parser.add_argument("--execute", action="store_true", help="Execute subruns. Without this flag, only chart_plan.json is produced.")
    parser.add_argument("--image-folder", default=None, help="Folder used for cases whose data_path is {image_folder}.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of cases to run.")
    parser.add_argument("--case-id", action="append", default=None, help="Run only selected case id. Can be repeated.")
    parser.add_argument("--fail-on-missing", action="store_true", help="Fail instead of skipping cases with missing input path.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the shared benchmark status bar.")
    return parser


def load_cases(path: str | Path) -> list[VirageE2ETestCase]:
    source = Path(path)
    cases: list[VirageE2ETestCase] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise ValueError("JSONL row must be an object.")
                cases.append(VirageE2ETestCase.from_dict(payload))
            except Exception as exc:
                raise ValueError(f"Invalid test case at {source}:{line_number}: {exc}") from exc
    return cases


def resolve_case_data_path(case: VirageE2ETestCase, *, image_folder: str | None, project_root: Path = PROJECT_ROOT) -> Path:
    raw_path = case.data_path
    if raw_path == "{image_folder}":
        if not image_folder:
            return Path(raw_path)
        raw_path = image_folder
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path)


def make_benchmark_run_id(value: str | None) -> str:
    if value:
        return value
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_virage_e2e_cases_" + uuid4().hex[:8]


def _safe_case_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")
    return safe or "case"


def _parse_orchestrator_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    start = text.rfind("{")
    if start < 0:
        return {}
    try:
        payload = json.loads(text[start:])
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_csv(path: Path, results: list[VirageE2ETestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].as_row().keys()) if results else [
        "case_id", "input_type", "data_path", "query", "status", "duration_seconds", "run_id", "run_dir",
        "chart_plan_path", "orchestrator_report_path", "subtask_count", "error", "stdout_path", "stderr_path",
        "focus", "expected_checks",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_row())


def _write_report(path: Path, *, run_id: str, cases_path: str, results: list[VirageE2ETestResult], execute: bool) -> None:
    total = len(results)
    completed = sum(1 for item in results if item.status == "completed")
    skipped = sum(1 for item in results if item.status.startswith("skipped"))
    failed = sum(1 for item in results if item.status == "failed")
    lines = [
        "# ViRAGE E2E test cases report",
        "",
        f"Run ID: `{run_id}`",
        f"Cases: `{cases_path}`",
        f"Mode: `{'execute' if execute else 'plan-only'}`",
        "",
        "| Status | Count |",
        "|---|---:|",
        f"| completed | {completed} |",
        f"| skipped | {skipped} |",
        f"| failed | {failed} |",
        f"| total | {total} |",
        "",
        "## Cases",
        "",
        "| Case | Status | Subtasks | Focus | Artifacts |",
        "|---|---|---:|---|---|",
    ]
    for item in results:
        artifacts = item.run_dir or item.error or ""
        focus = ", ".join(item.focus)
        subtasks = "" if item.subtask_count is None else str(item.subtask_count)
        lines.append(
            f"| `{item.case_id}` | {item.status} | {subtasks} | {focus.replace('|', '/')} | `{str(artifacts).replace('|', '/')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_case(
        case: VirageE2ETestCase,
        *,
        config_path: str,
        benchmark_run_id: str,
        output_dir: Path,
        artifact_root: str | None,
        execute: bool,
        image_folder: str | None,
        fail_on_missing: bool,
) -> VirageE2ETestResult:
    resolved_data_path = resolve_case_data_path(case, image_folder=image_folder)
    if not resolved_data_path.exists():
        message = f"Input path does not exist: {resolved_data_path}"
        if fail_on_missing:
            return VirageE2ETestResult(
                case_id=case.case_id,
                input_type=case.input_type,
                data_path=resolved_data_path.as_posix(),
                query=case.query,
                status="failed",
                error=message,
                focus=case.focus,
                expected_checks=case.expected_checks,
            )
        return VirageE2ETestResult(
            case_id=case.case_id,
            input_type=case.input_type,
            data_path=resolved_data_path.as_posix(),
            query=case.query,
            status="skipped_missing_input",
            error=message,
            focus=case.focus,
            expected_checks=case.expected_checks,
        )

    safe_id = _safe_case_id(case.case_id)
    case_run_id = f"{benchmark_run_id}/{safe_id}"
    stdout_path = output_dir / "stdout" / f"{safe_id}.stdout.txt"
    stderr_path = output_dir / "stderr" / f"{safe_id}.stderr.txt"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_orchestrator.py"),
        "--config",
        config_path,
        "--query",
        case.query,
        "--data-path",
        resolved_data_path.as_posix(),
        "--input-type",
        case.input_type,
        "--run-id",
        case_run_id,
    ]
    if artifact_root:
        command.extend(["--artifact-root", artifact_root])
    if execute:
        command.append("--execute")

    start = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    duration = time.perf_counter() - start
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    payload = _parse_orchestrator_stdout(completed.stdout or "")
    if completed.returncode != 0:
        return VirageE2ETestResult(
            case_id=case.case_id,
            input_type=case.input_type,
            data_path=resolved_data_path.as_posix(),
            query=case.query,
            status="failed",
            duration_seconds=round(duration, 6),
            run_id=case_run_id,
            error=(completed.stderr or completed.stdout or f"exit code {completed.returncode}").strip()[:2000],
            stdout_path=stdout_path.as_posix(),
            stderr_path=stderr_path.as_posix(),
            focus=case.focus,
            expected_checks=case.expected_checks,
        )
    return VirageE2ETestResult(
        case_id=case.case_id,
        input_type=case.input_type,
        data_path=resolved_data_path.as_posix(),
        query=case.query,
        status="completed",
        duration_seconds=round(duration, 6),
        run_id=str(payload.get("run_id") or case_run_id),
        run_dir=payload.get("run_dir"),
        chart_plan_path=payload.get("chart_plan"),
        orchestrator_report_path=payload.get("orchestrator_report"),
        subtask_count=payload.get("subtask_count"),
        stdout_path=stdout_path.as_posix(),
        stderr_path=stderr_path.as_posix(),
        focus=case.focus,
        expected_checks=case.expected_checks,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = make_benchmark_run_id(args.run_id)
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "artifacts" / run_id / "e2e_cases_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case.case_id in selected]
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]

    _write_json(output_dir / "benchmark_request.json", {
        "run_id": run_id,
        "cases_path": str(args.cases),
        "config": args.config,
        "execute": bool(args.execute),
        "case_count": len(cases),
        "image_folder": args.image_folder,
    })

    progress = ConsoleProgressBar(total=len(cases), title="ViRAGE E2E cases", enabled=not args.no_progress)
    results: list[VirageE2ETestResult] = []
    ok_count = 0
    error_count = 0
    skipped_count = 0
    for index, case in enumerate(cases, start=1):
        progress.update(index - 1, ok=ok_count, errors=error_count, reused=skipped_count, stage="orchestrator", label=case.case_id)
        result = _run_case(
            case,
            config_path=args.config,
            benchmark_run_id=run_id,
            output_dir=output_dir,
            artifact_root=args.artifact_root,
            execute=bool(args.execute),
            image_folder=args.image_folder,
            fail_on_missing=bool(args.fail_on_missing),
        )
        results.append(result)
        if result.status == "completed":
            ok_count += 1
        elif result.status.startswith("skipped"):
            skipped_count += 1
        else:
            error_count += 1
        progress.update(index, ok=ok_count, errors=error_count, reused=skipped_count, stage="orchestrator", label=case.case_id)
    progress.close(label="completed")

    result_rows = [item.as_row() for item in results]
    output_files = {
        "per_case_csv": (output_dir / "per_case_results.csv").as_posix(),
        "per_case_jsonl": (output_dir / "per_case_results.jsonl").as_posix(),
        "summary_json": (output_dir / "benchmark_summary.json").as_posix(),
        "report_md": (output_dir / "benchmark_report.md").as_posix(),
    }
    summary = {
        "run_id": run_id,
        "cases_path": str(args.cases),
        "execute": bool(args.execute),
        "total_cases": len(results),
        "completed_cases": ok_count,
        "failed_cases": error_count,
        "skipped_cases": skipped_count,
        "output_dir": output_dir.as_posix(),
        "output_files": output_files,
    }
    _write_csv(output_dir / "per_case_results.csv", results)
    _write_jsonl(output_dir / "per_case_results.jsonl", result_rows)
    _write_json(output_dir / "benchmark_summary.json", summary)
    _write_report(output_dir / "benchmark_report.md", run_id=run_id, cases_path=str(args.cases), results=results, execute=bool(args.execute))
    print(json.dumps({"status": "completed", **summary}, ensure_ascii=False, indent=2))
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
