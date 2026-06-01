from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.project_config import load_project_config


def _load_user_context(value: str | None, file_path: str | None) -> dict[str, Any]:
    if value and file_path:
        raise ValueError("Use either --user-context-json or --user-context-file, not both.")
    if not value and not file_path:
        return {}
    raw = Path(file_path).read_text(encoding="utf-8") if file_path else str(value)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("User context must be a JSON object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one ViRAGE pipeline request and save reproducible artifacts.")
    parser.add_argument("--config", required=True, help="Path to project TOML config.")
    parser.add_argument("--query", required=True, help="User analytical request.")
    parser.add_argument("--data-path", required=True, help="Path to input CSV/XLSX table.")
    parser.add_argument("--run-id", default=None, help="Optional deterministic run id.")
    parser.add_argument("--artifact-root", default=None, help="Override settings.artifact_root for this run.")
    parser.add_argument("--user-context-json", default=None, help="Optional JSON object with extra user context.")
    parser.add_argument("--user-context-file", default=None, help="Optional path to JSON object with extra user context.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_project_config(args.config)
    if args.artifact_root:
        config.settings.artifact_root = Path(args.artifact_root)
    request = PipelineRequest(
        query=args.query,
        data_path=args.data_path,
        user_context=_load_user_context(args.user_context_json, args.user_context_file),
        **({"run_id": args.run_id} if args.run_id else {}),
    )

    pipeline = ViRAGEPipeline.from_project_config(config)
    result = pipeline.invoke(request)
    run_dir = pipeline.runtime.ensure_run_dir(request.run_id)
    summary = {
        "run_id": result.run_id,
        "status": "completed",
        "run_dir": run_dir.as_posix(),
        "run_report": (run_dir / "run_report.json").as_posix(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
