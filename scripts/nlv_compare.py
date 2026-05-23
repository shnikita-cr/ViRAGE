import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_METRICS = [
    "total_cases",
    "successful_cases",
    "failed_cases",
    "visualization_error_rate",
    "empty_chart_rate",
    "chart_text_consistency_rate",
    "mean_spec_score",
    "mean_spec_score_failure_as_zero",
    "mean_vision_score",
    "mean_vision_score_failure_as_zero",
    "median_spec_score",
    "median_vision_score",
    "mean_duration_seconds",
    "p95_duration_seconds",
    "total_tokens",
    "mean_total_tokens",
]

HIGHER_IS_BETTER = {
    "successful_cases",
    "chart_text_consistency_rate",
    "mean_spec_score",
    "mean_spec_score_failure_as_zero",
    "mean_vision_score",
    "mean_vision_score_failure_as_zero",
    "median_spec_score",
    "median_vision_score",
}

LOWER_IS_BETTER = {
    "failed_cases",
    "visualization_error_rate",
    "empty_chart_rate",
    "mean_duration_seconds",
    "p95_duration_seconds",
    "total_tokens",
    "mean_total_tokens",
}


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_direction(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    return "neutral"


def improvement_value(metric: str, no_rag_value, rag_value):
    if not isinstance(no_rag_value, (int, float)) or not isinstance(rag_value, (int, float)):
        return None

    raw_delta = rag_value - no_rag_value

    if metric in LOWER_IS_BETTER:
        return round(-raw_delta, 6)
    if metric in HIGHER_IS_BETTER:
        return round(raw_delta, 6)
    return None


def build_dataframe(no_rag: dict, rag: dict, metrics: list[str]) -> pd.DataFrame:
    rows = []

    for metric in metrics:
        no_rag_value = no_rag.get(metric)
        rag_value = rag.get(metric)

        raw_delta = None
        if isinstance(no_rag_value, (int, float)) and isinstance(rag_value, (int, float)):
            raw_delta = round(rag_value - no_rag_value, 6)

        improvement = improvement_value(metric, no_rag_value, rag_value)

        if improvement is None:
            status = ""
        elif improvement > 0:
            status = "better"
        elif improvement < 0:
            status = "worse"
        else:
            status = "same"

        rows.append(
            {
                "metric": metric,
                "no_rag": no_rag_value,
                "rag": rag_value,
                "raw_delta_rag_minus_no_rag": raw_delta,
                "direction": metric_direction(metric),
                "improvement": improvement,
                "status": status,
            }
        )

    return pd.DataFrame(rows)


def format_value(value):
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def print_dataframe(df: pd.DataFrame) -> None:
    printable = df.copy()
    for column in ["no_rag", "rag", "raw_delta_rag_minus_no_rag", "improvement"]:
        printable[column] = printable[column].map(format_value)

    with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            220,
            "display.max_colwidth",
            80,
    ):
        print(printable.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty NLV no-RAG vs RAG benchmark comparison.")
    parser.add_argument(
        "--no-rag-report",
        type=Path,
        default=Path("artifacts/benchmarks/nlv_no_rag/benchmark_report.json"),
        help="Path to no-RAG benchmark_report.json.",
    )
    parser.add_argument(
        "--rag-report",
        type=Path,
        default=Path("artifacts/benchmarks/nlv_rag/benchmark_report.json"),
        help="Path to RAG benchmark_report.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/benchmarks/nlv_compare"),
        help="Directory for comparison outputs.",
    )

    args = parser.parse_args()

    no_rag = load_report(args.no_rag_report)
    rag = load_report(args.rag_report)

    df = build_dataframe(no_rag=no_rag, rag=rag, metrics=DEFAULT_METRICS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "nlv_compare_metrics.csv", index=False, encoding="utf-8")
    df.to_json(args.output_dir / "nlv_compare_metrics.json", orient="records", force_ascii=False, indent=2)

    print_dataframe(df)


if __name__ == "__main__":
    main()
