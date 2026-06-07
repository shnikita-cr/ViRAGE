from __future__ import annotations

import argparse
import site
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_ROOT.as_posix())
from typing import Any


def main() -> None:
    args = parse_args()
    rows = _collect_rows(Path(args.benchmark_report), limit=args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    print(json.dumps({"items": len(rows), "output": output.as_posix()}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export query+image rows for manual 0/1/2 semantic review.")
    parser.add_argument("--benchmark-report", required=True, help="benchmark_report.json from NLV/external benchmark.")
    parser.add_argument("--output", default="artifacts/benchmarks/manual_semantic_review/review_set.csv")
    parser.add_argument("--limit", type=int, default=50)
    return parser.parse_args()


def _collect_rows(report_path: Path, *, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    rows: list[dict[str, Any]] = []
    for item in results[: max(0, limit)]:
        rows.append({
            "case_id": item.get("case_id", ""),
            "dataset_name": item.get("dataset_name", ""),
            "query": item.get("query", ""),
            "image_path": item.get("generated_image_path") or item.get("rendered_image_path") or item.get("image_path") or "",
            "run_id": item.get("run_id", ""),
            "chart_type": (item.get("metadata") or {}).get("chart_type", ""),
            "vlm_judge_score": item.get("vlm_judge_score", ""),
            "embedding_score": item.get("embedding_score", ""),
            "semantic_match_score": item.get("semantic_match_score", ""),
            "manual_score": "",
            "manual_comment": "",
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["case_id", "dataset_name", "query", "image_path", "run_id", "chart_type", "vlm_judge_score", "embedding_score", "semantic_match_score", "manual_score", "manual_comment"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
