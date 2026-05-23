from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_NUMERIC_METRICS = (
    "spec_score",
    "vision_score",
    "duration_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
_RESULT_FILE_CANDIDATES = (
    "benchmark_report.json",
    "analysis_benchmark_report.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two ViRAGE benchmark runs by paired case_id.")
    parser.add_argument("--left", required=True, help="Baseline run directory, for example no RAG.")
    parser.add_argument("--right", required=True, help="Candidate run directory, for example RAG.")
    parser.add_argument("--output", required=True, help="Output directory for comparison report.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples for CI.")
    args = parser.parse_args()

    left = _load_run(Path(args.left))
    right = _load_run(Path(args.right))
    comparison = compare_runs(left, right, bootstrap_samples=max(0, args.bootstrap))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_outputs(comparison, output)
    print(f"Common cases: {comparison['summary']['common_cases']}")
    print(f"Improved: {comparison['summary']['improved_cases']}")
    print(f"Degraded: {comparison['summary']['degraded_cases']}")
    print(f"Report directory: {output.resolve()}")


def compare_runs(left: dict, right: dict, *, bootstrap_samples: int) -> dict:
    left_items = {str(item.get("case_id")): item for item in left["results"] if item.get("case_id")}
    right_items = {str(item.get("case_id")): item for item in right["results"] if item.get("case_id")}
    common_ids = sorted(set(left_items) & set(right_items))
    rows = []
    improved = degraded = fixed = broken = unchanged = 0
    for case_id in common_ids:
        l_item = left_items[case_id]
        r_item = right_items[case_id]
        row = _compare_case(case_id, l_item, r_item)
        rows.append(row)
        quality_delta = row.get("delta_spec_score_failure_as_zero")
        if isinstance(quality_delta, (int, float)):
            if quality_delta > 1e-9:
                improved += 1
            elif quality_delta < -1e-9:
                degraded += 1
            else:
                unchanged += 1
        if _is_failed(l_item) and not _is_failed(r_item):
            fixed += 1
        if not _is_failed(l_item) and _is_failed(r_item):
            broken += 1
    metric_deltas = _aggregate_metric_deltas(rows, bootstrap_samples=bootstrap_samples)
    return {
        "summary": {
            "left_cases": len(left_items),
            "right_cases": len(right_items),
            "common_cases": len(common_ids),
            "missing_in_left": sorted(set(right_items) - set(left_items)),
            "missing_in_right": sorted(set(left_items) - set(right_items)),
            "improved_cases": improved,
            "degraded_cases": degraded,
            "unchanged_cases": unchanged,
            "fixed_by_right": fixed,
            "broken_by_right": broken,
        },
        "left_manifest": left.get("manifest"),
        "right_manifest": right.get("manifest"),
        "metric_deltas": metric_deltas,
        "rows": rows,
    }


def _compare_case(case_id: str, left: dict, right: dict) -> dict:
    row = {
        "case_id": case_id,
        "left_error": left.get("error"),
        "right_error": right.get("error"),
        "left_failed": _is_failed(left),
        "right_failed": _is_failed(right),
    }
    for metric in _NUMERIC_METRICS:
        l_value = _number_or_none(left.get(metric))
        r_value = _number_or_none(right.get(metric))
        row[f"left_{metric}"] = l_value
        row[f"right_{metric}"] = r_value
        row[f"delta_{metric}"] = None if l_value is None or r_value is None else round(r_value - l_value, 6)
        l_zero = l_value if l_value is not None and not _is_failed(left) else 0.0
        r_zero = r_value if r_value is not None and not _is_failed(right) else 0.0
        row[f"left_{metric}_failure_as_zero"] = l_zero
        row[f"right_{metric}_failure_as_zero"] = r_zero
        row[f"delta_{metric}_failure_as_zero"] = round(r_zero - l_zero, 6)
    return row


def _aggregate_metric_deltas(rows: list[dict], *, bootstrap_samples: int) -> dict:
    out = {}
    for key in sorted({key for row in rows for key in row if key.startswith("delta_")}):
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        out[key] = {
            "mean": round(sum(values) / len(values), 6),
            "median": round(statistics.median(values), 6),
            "ci95": _bootstrap_ci(values, samples=bootstrap_samples),
        }
    return out


def _bootstrap_ci(values: list[float], *, samples: int) -> dict[str, float] | None:
    if not values or samples <= 0:
        return None
    rng = random.Random(42)
    means = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]
    return {"low": round(low, 6), "high": round(high, 6)}


def _load_run(path: Path) -> dict:
    report_path = next((path / name for name in _RESULT_FILE_CANDIDATES if (path / name).exists()), None)
    if report_path is None:
        raise FileNotFoundError(f"No benchmark report found in {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_path = path / "benchmark_run_manifest.json"
    return {
        "report": report,
        "results": report.get("results", []),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None,
    }


def _write_outputs(comparison: dict, output: Path) -> None:
    (output / "comparison_report.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = comparison["rows"]
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (output / "comparison_cases.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Benchmark comparison",
        "",
        f"Common cases: {comparison['summary']['common_cases']}",
        f"Improved cases: {comparison['summary']['improved_cases']}",
        f"Degraded cases: {comparison['summary']['degraded_cases']}",
        f"Fixed by right: {comparison['summary']['fixed_by_right']}",
        f"Broken by right: {comparison['summary']['broken_by_right']}",
        "",
        "## Metric deltas",
        "",
    ]
    for key, value in comparison["metric_deltas"].items():
        lines.append(f"- {key}: mean={value['mean']}, median={value['median']}, ci95={value['ci95']}")
    (output / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def _is_failed(item: dict) -> bool:
    return bool(item.get("error"))


def _number_or_none(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
