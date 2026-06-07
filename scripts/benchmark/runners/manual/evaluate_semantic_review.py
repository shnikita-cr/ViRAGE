from __future__ import annotations

import argparse
import csv
import json
import math
import site
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_ROOT.as_posix())

METRIC_FIELDS = ("semantic_match_score", "vlm_judge_score", "embedding_score")


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.input))
    scored = [_score_row(row, threshold=args.threshold) for row in rows]
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
    parser.add_argument("--threshold", type=float, default=0.7, help="Automatic metric threshold for binary good/bad.")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _score_row(row: dict[str, Any], *, threshold: float) -> dict[str, Any]:
    manual = _float(row.get("manual_score"))
    payload: dict[str, Any] = {
        **row,
        "manual_score_valid": int(manual is not None),
        "manual_good": int(manual is not None and manual >= 2.0),
    }
    for metric in METRIC_FIELDS:
        payload[f"{metric}.good"] = int((_float(row.get(metric)) or 0.0) >= threshold)
    return payload


def _report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if int(row.get("manual_score_valid") or 0) == 1]
    return {
        "items": len(rows),
        "valid_manual_items": len(valid_rows),
        "mean_manual_score": _mean_column(valid_rows, "manual_score"),
        "metrics": {metric: _metric_report(valid_rows, metric) for metric in METRIC_FIELDS},
    }


def _metric_report(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return {
        "mean_score": _mean_column(rows, metric),
        "spearman_correlation": _spearman(rows, x_field="manual_score", y_field=metric),
        "binary": _binary_report(rows, predicted_field=f"{metric}.good"),
    }


def _binary_report(rows: list[dict[str, Any]], *, predicted_field: str) -> dict[str, Any]:
    pairs = [(int(row["manual_good"]), int(row[predicted_field])) for row in rows if row.get(predicted_field) not in (None, "")]
    tp = sum(1 for expected, actual in pairs if expected == 1 and actual == 1)
    tn = sum(1 for expected, actual in pairs if expected == 0 and actual == 0)
    fp = sum(1 for expected, actual in pairs if expected == 0 and actual == 1)
    fn = sum(1 for expected, actual in pairs if expected == 1 and actual == 0)
    total = len(pairs)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "items": total,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _spearman(rows: list[dict[str, Any]], *, x_field: str, y_field: str) -> float | None:
    pairs = [(x, y) for row in rows if (x := _float(row.get(x_field))) is not None and (y := _float(row.get(y_field))) is not None]
    if len(pairs) < 2:
        return None
    x_ranks = _average_ranks([pair[0] for pair in pairs])
    y_ranks = _average_ranks([pair[1] for pair in pairs])
    return _pearson(x_ranks, y_ranks)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return numerator / (denom_x * denom_y)


def _mean_column(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [value for row in rows if (value := _float(row.get(field))) is not None]
    return mean(values) if values else None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return 2.0 * precision * recall / (precision + recall)


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
