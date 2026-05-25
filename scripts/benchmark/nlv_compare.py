from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_METRICS = [
    "total_cases",
    "successful_cases",
    "failed_cases",
    "visualization_error_rate",
    "empty_chart_rate",
    "mean_spec_score",
    "mean_spec_score_failure_as_zero",
    "mean_vision_score",
    "mean_vision_score_failure_as_zero",
    "median_spec_score",
    "median_vision_score",
    "mean_duration_seconds",
    "p95_duration_seconds",
    "mean_prompt_tokens",
    "mean_completion_tokens",
    "mean_total_tokens",
    "total_tokens",
    "chart_text_consistency_rate",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return None


def build_dataframe(no_rag: dict[str, Any], rag: dict[str, Any], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        a = no_rag.get(metric)
        b = rag.get(metric)
        a_num = _number(a)
        b_num = _number(b)
        rows.append({
            "metric": metric,
            "no_rag": a,
            "rag": b,
            "delta_rag_minus_no_rag": None if a_num is None or b_num is None else round(float(b_num) - float(a_num), 6),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print and save a compact NLV no-RAG vs RAG metrics table.")
    parser.add_argument("--no-rag-report", default="artifacts/benchmarks/nlv_no_rag/benchmark_report.json")
    parser.add_argument("--rag-report", default="artifacts/benchmarks/nlv_rag/benchmark_report.json")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/nlv_compare")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()

    no_rag_report = Path(args.no_rag_report)
    rag_report = Path(args.rag_report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(_read_json(no_rag_report), _read_json(rag_report), list(args.metrics))
    print(df.to_string(index=False))
    df.to_csv(output_dir / "nlv_compare_metrics.csv", index=False)
    df.to_json(output_dir / "nlv_compare_metrics.json", orient="records", force_ascii=False, indent=2)


if __name__ == "__main__":
    main()
