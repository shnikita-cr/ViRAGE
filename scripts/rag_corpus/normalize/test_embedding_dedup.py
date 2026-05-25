from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in _CURRENT_FILE.parents if (parent / "src").exists() and (parent / "scripts").exists()), Path.cwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag_corpus.normalize.deduplicate_by_embeddings import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_records,
    cosine,
    deduplicate,
    embed_records,
    l2_normalize,
    ollama_embed,
    write_jsonl,
)


@dataclass(frozen=True)
class PairCheck:
    name: str
    left_doc_id: str
    right_doc_id: str
    expected: str
    similarity: float


SAMPLE_RECORDS: list[dict[str, Any]] = [
    {
        "doc_id": "sample_line_trend_a",
        "record_type": "chart_pattern",
        "chart_family": "line_chart",
        "title": "Line chart for temporal trends",
        "prompt_text": "Use a line chart when the user wants to show how a quantitative value changes over time. Put the temporal field on the x-axis and the quantitative measure on the y-axis.",
        "retrieval_text": "time trend line chart temporal quantitative measure",
    },
    {
        "doc_id": "sample_line_trend_b",
        "record_type": "chart_pattern",
        "chart_family": "line_chart",
        "title": "Temporal trend line plot",
        "prompt_text": "For a trend over time, choose a line plot with time on the horizontal axis and the numeric value on the vertical axis.",
        "retrieval_text": "line plot for values over time trend",
    },
    {
        "doc_id": "sample_scatter_relationship",
        "record_type": "chart_pattern",
        "chart_family": "scatter_plot",
        "title": "Scatter plot for two quantitative fields",
        "prompt_text": "Use a scatter plot when the task is to inspect the relationship between two quantitative fields. Encode one numeric field on x and the other numeric field on y.",
        "retrieval_text": "scatter plot relationship correlation two quantitative fields",
    },
    {
        "doc_id": "sample_bar_aggregate",
        "record_type": "chart_pattern",
        "chart_family": "bar_chart",
        "title": "Bar chart for aggregated category comparison",
        "prompt_text": "Use a bar chart when the task is to compare an aggregated quantitative measure across categories. Put the category on one axis and the aggregate measure on the other axis.",
        "retrieval_text": "bar chart compare aggregate measure by category",
    },
]

PAIR_DEFINITIONS = [
    ("similar_line_rules", "sample_line_trend_a", "sample_line_trend_b", "duplicate"),
    ("line_vs_scatter", "sample_line_trend_a", "sample_scatter_relationship", "different"),
    ("line_vs_bar", "sample_line_trend_a", "sample_bar_aggregate", "different"),
    ("scatter_vs_bar", "sample_scatter_relationship", "sample_bar_aggregate", "different"),
]


def vector_text(record: dict[str, Any]) -> str:
    return "\n".join(
        str(record.get(key) or "").strip()
        for key in ["title", "prompt_text", "retrieval_text", "chart_family", "record_type"]
        if str(record.get(key) or "").strip()
    )


def compute_pair_checks(records: list[dict[str, Any]], model: str, base_url: str, timeout: float, retries: int) -> list[PairCheck]:
    by_id = {str(record["doc_id"]): record for record in records}
    vectors: dict[str, list[float]] = {}
    for doc_id, record in by_id.items():
        vectors[doc_id] = l2_normalize(ollama_embed(vector_text(record), model, base_url, timeout, retries))

    checks: list[PairCheck] = []
    for name, left, right, expected in PAIR_DEFINITIONS:
        checks.append(
            PairCheck(
                name=name,
                left_doc_id=left,
                right_doc_id=right,
                expected=expected,
                similarity=round(float(cosine(vectors[left], vectors[right])), 6),
            )
        )
    return checks


def run_threshold_sweep(
    records: list[dict[str, Any]],
    model: str,
    base_url: str,
    thresholds: list[float],
    timeout: float,
    retries: int,
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="virage_embedding_dedup_test_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "sample.jsonl"
        cache_path = tmp_path / "cache.jsonl"
        write_jsonl(input_path, records)

        corpus_records, skipped = build_records(
            records,
            fields=["title", "prompt_text", "retrieval_text", "chart_family", "record_type"],
            group_mode="record_type",
            min_text_chars=10,
        )
        if skipped:
            raise RuntimeError(f"Sample records should not be skipped, got: {skipped}")

        embeddings = embed_records(corpus_records, model, base_url, timeout, retries, cache_path)
        rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            keep_indexes, duplicates, skipped_groups = deduplicate(
                records=corpus_records,
                embeddings=embeddings,
                threshold=threshold,
                max_group_size=100,
            )
            removed_ids = {decision.removed_doc_id for decision in duplicates}
            expected_duplicate_removed = "sample_line_trend_b" in removed_ids
            different_removed = sorted(
                removed_id
                for removed_id in removed_ids
                if removed_id in {"sample_scatter_relationship", "sample_bar_aggregate"}
            )
            rows.append(
                {
                    "threshold": threshold,
                    "kept_records": len(keep_indexes),
                    "removed_records": len(duplicates),
                    "expected_duplicate_removed": expected_duplicate_removed,
                    "different_records_removed": different_removed,
                    "status": "PASS" if expected_duplicate_removed and not different_removed else "WARN",
                    "duplicates": [decision.__dict__ for decision in duplicates],
                    "skipped_groups": skipped_groups,
                }
            )
        return rows


def recommend_threshold(sweep_rows: list[dict[str, Any]]) -> float | None:
    passing = [row for row in sweep_rows if row["status"] == "PASS"]
    if not passing:
        return None
    # Prefer the most conservative passing threshold.
    return max(float(row["threshold"]) for row in passing)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that Ollama embeddings can separate duplicate and different ViRAGE RAG rules.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98])
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("rag_corpus/processed/embedding_dedup_test"))
    args = parser.parse_args()

    checks = compute_pair_checks(SAMPLE_RECORDS, args.model, args.base_url, args.timeout, args.retries)
    sweep_rows = run_threshold_sweep(SAMPLE_RECORDS, args.model, args.base_url, args.thresholds, args.timeout, args.retries)
    recommended = recommend_threshold(sweep_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "base_url": args.base_url,
        "pair_similarities": [check.__dict__ for check in checks],
        "threshold_sweep": sweep_rows,
        "recommended_threshold": recommended,
        "interpretation": {
            "PASS": "The known duplicate is removed and known different records are kept.",
            "WARN": "Either the known duplicate remained or at least one different record was removed.",
        },
    }
    (args.output_dir / "embedding_dedup_test_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Embedding deduplication test", "", f"Model: `{args.model}`", "", "## Pair similarities", ""]
    md.append("| Pair | Expected | Similarity |")
    md.append("|---|---:|---:|")
    for check in checks:
        md.append(f"| {check.name} | {check.expected} | {check.similarity:.6f} |")
    md.extend(["", "## Threshold sweep", "", "| Threshold | Status | Removed | Expected duplicate removed | Different records removed |", "|---:|---|---:|---:|---|"])
    for row in sweep_rows:
        diff = ", ".join(row["different_records_removed"]) or "-"
        md.append(f"| {row['threshold']:.2f} | {row['status']} | {row['removed_records']} | {row['expected_duplicate_removed']} | {diff} |")
    md.extend(["", f"Recommended threshold: `{recommended}`" if recommended is not None else "Recommended threshold: none", ""])
    (args.output_dir / "embedding_dedup_test_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({"recommended_threshold": recommended, "pair_similarities": [check.__dict__ for check in checks]}, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_dir / 'embedding_dedup_test_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
