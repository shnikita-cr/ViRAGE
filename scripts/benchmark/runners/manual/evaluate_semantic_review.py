from __future__ import annotations

import argparse
import site
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_ROOT.as_posix())
from statistics import mean
from typing import Any


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.input))
    scored = [_score_row(row) for row in rows]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "manual_semantic_review_scored.csv", scored)
    report = _report(scored)
    (output / "manual_semantic_review_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate manual 0/1/2 review against semantic metrics.")
    parser.add_argument("--input", required=True, help="CSV with manual_score and automatic metric columns.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/manual_semantic_review")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    manual = _float(row.get("manual_score"))
    return {
        **row,
        "manual_good": int(manual is not None and manual >= 2.0),
        "semantic_good": int((_float(row.get("semantic_match_score")) or 0.0) >= 0.7),
        "vlm_good": int((_float(row.get("vlm_judge_score")) or 0.0) >= 0.7),
        "embedding_good": int((_float(row.get("embedding_score")) or 0.0) >= 0.7),
    }


def _report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": len(rows),
        "semantic_accuracy": _binary_accuracy(rows, "semantic_good"),
        "vlm_accuracy": _binary_accuracy(rows, "vlm_good"),
        "embedding_accuracy": _binary_accuracy(rows, "embedding_good"),
        "mean_manual_score": _mean_column(rows, "manual_score"),
        "mean_semantic_match_score": _mean_column(rows, "semantic_match_score"),
        "mean_vlm_judge_score": _mean_column(rows, "vlm_judge_score"),
        "mean_embedding_score": _mean_column(rows, "embedding_score"),
    }


def _binary_accuracy(rows: list[dict[str, Any]], field: str) -> float | None:
    pairs = [(int(row["manual_good"]), int(row[field])) for row in rows if row.get("manual_score") not in (None, "")]
    if not pairs:
        return None
    return sum(1 for expected, actual in pairs if expected == actual) / len(pairs)


def _mean_column(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [value for row in rows if (value := _float(row.get(field))) is not None]
    return mean(values) if values else None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
