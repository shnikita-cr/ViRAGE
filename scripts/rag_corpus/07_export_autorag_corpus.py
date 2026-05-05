from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_JSONL = "rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl"
DEFAULT_OUT_DIR = "rag_corpus/autorag/vega_lite"
DEFAULT_CORPUS_FILE = "corpus.parquet"
DEFAULT_REPORT_FILE = "corpus_export_report.md"
DEFAULT_REPORT_JSON_FILE = "corpus_export_report.json"

DEFAULT_METADATA_FIELDS = [
    "source",
    "source_path",
    "file_name",
    "file_stem",
    "title",
    "source_split",
    "benchmark_group",
    "is_eval_leak_sensitive",
    "corpus_type",
    "mark_type",
    "chart_pattern",
    "field_roles",
    "data_policy",
    "removed_data_sections",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL file does not exist: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc

            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at line {line_number}")

            records.append(value)

    return records


def as_metadata_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def build_metadata(
    record: dict[str, Any],
    metadata_fields: list[str],
    generated_at: datetime,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "last_modified_datetime": generated_at,
    }

    for field in metadata_fields:
        if field not in record:
            continue

        metadata[field] = as_metadata_value(record[field])

    return metadata


def build_corpus_rows(
    records: list[dict[str, Any]],
    content_field: str,
    doc_id_field: str,
    metadata_fields: list[str],
    generated_at: datetime,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        doc_id = record.get(doc_id_field)
        contents = record.get(content_field)

        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"Record #{index} has invalid {doc_id_field}: {doc_id!r}")

        if not isinstance(contents, str) or not contents.strip():
            raise ValueError(f"Record {doc_id} has invalid {content_field}: {contents!r}")

        source_path = record.get("source_path")
        path = source_path if isinstance(source_path, str) else ""

        rows.append(
            {
                "doc_id": doc_id,
                "contents": contents,
                "path": path,
                "metadata": build_metadata(
                    record=record,
                    metadata_fields=metadata_fields,
                    generated_at=generated_at,
                ),
            }
        )

    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    doc_ids = [row["doc_id"] for row in rows]
    duplicates = [doc_id for doc_id, count in Counter(doc_ids).items() if count > 1]

    if duplicates:
        preview = ", ".join(duplicates[:20])
        raise ValueError(f"Duplicate doc_id values found: {preview}")

    for row in rows:
        if not isinstance(row["doc_id"], str):
            raise ValueError(f"doc_id must be string: {row!r}")

        if not isinstance(row["contents"], str):
            raise ValueError(f"contents must be string: {row!r}")

        if not isinstance(row["metadata"], dict):
            raise ValueError(f"metadata must be dict: {row!r}")

        if "last_modified_datetime" not in row["metadata"]:
            raise ValueError(f"metadata.last_modified_datetime is required: {row!r}")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.reset_index(drop=True)

    dataframe.to_parquet(path, index=False)


def read_back_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def count_field(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()

    for record in records:
        value = record.get(field)

        if isinstance(value, str):
            counter[value] += 1
        elif value is None:
            counter["[null]"] += 1
        else:
            counter[str(value)] += 1

    return dict(counter.most_common())


def build_report_data(
    input_jsonl: Path,
    corpus_path: Path,
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    content_field: str,
    doc_id_field: str,
    metadata_fields: list[str],
) -> dict[str, Any]:
    content_lengths = [len(row["contents"]) for row in rows]

    return {
        "input_jsonl": input_jsonl.as_posix(),
        "corpus_path": corpus_path.as_posix(),
        "record_count": len(records),
        "row_count": len(rows),
        "content_field": content_field,
        "doc_id_field": doc_id_field,
        "metadata_fields": metadata_fields,
        "min_content_length": min(content_lengths) if content_lengths else 0,
        "max_content_length": max(content_lengths) if content_lengths else 0,
        "avg_content_length": round(sum(content_lengths) / len(content_lengths), 2)
        if content_lengths
        else 0,
        "source_distribution": count_field(records, "source"),
        "corpus_type_distribution": count_field(records, "corpus_type"),
        "chart_pattern_distribution": count_field(records, "chart_pattern"),
        "mark_type_distribution": count_field(records, "mark_type"),
        "sample_rows": [
            {
                "doc_id": row["doc_id"],
                "path": row["path"],
                "contents": row["contents"][:400],
                "metadata": {
                    key: value
                    for key, value in row["metadata"].items()
                    if key != "last_modified_datetime"
                },
            }
            for row in rows[:10]
        ],
    }


def markdown_counter(title: str, values: dict[str, int]) -> list[str]:
    lines: list[str] = []

    lines.append(f"## {title}")
    lines.append("")

    if not values:
        lines.append("_No data._")
        lines.append("")
        return lines

    lines.append("| Value | Count |")
    lines.append("|---|---:|")

    for value, count in values.items():
        lines.append(f"| `{value}` | {count} |")

    lines.append("")
    return lines


def build_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# AutoRAG corpus export report")
    lines.append("")
    lines.append(f"Input JSONL: `{report['input_jsonl']}`")
    lines.append(f"Corpus parquet: `{report['corpus_path']}`")
    lines.append(f"Records: **{report['record_count']}**")
    lines.append(f"Rows: **{report['row_count']}**")
    lines.append(f"Content field: `{report['content_field']}`")
    lines.append(f"Doc id field: `{report['doc_id_field']}`")
    lines.append(f"Min content length: `{report['min_content_length']}`")
    lines.append(f"Max content length: `{report['max_content_length']}`")
    lines.append(f"Avg content length: `{report['avg_content_length']}`")
    lines.append("")

    lines.extend(markdown_counter("Source distribution", report["source_distribution"]))
    lines.extend(markdown_counter("Corpus type distribution", report["corpus_type_distribution"]))
    lines.extend(markdown_counter("Chart pattern distribution", report["chart_pattern_distribution"]))
    lines.extend(markdown_counter("Mark type distribution", report["mark_type_distribution"]))

    lines.append("## Sample rows")
    lines.append("")
    lines.append("| doc_id | path | contents preview |")
    lines.append("|---|---|---|")

    for row in report["sample_rows"]:
        preview = row["contents"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{row['doc_id']}` | `{row['path']}` | {preview} |")

    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: Path, md_name: str, json_name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / md_name
    json_path = out_dir / json_name

    md_path.write_text(build_markdown_report(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return md_path, json_path


def parse_metadata_fields(raw_value: str) -> list[str]:
    value = raw_value.strip()

    if not value:
        return DEFAULT_METADATA_FIELDS

    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export normalized JSONL records to AutoRAG corpus.parquet."
    )

    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Path to normalized JSONL file.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for AutoRAG corpus files.",
    )
    parser.add_argument(
        "--corpus-file",
        default=DEFAULT_CORPUS_FILE,
        help="Output corpus parquet filename.",
    )
    parser.add_argument(
        "--report-file",
        default=DEFAULT_REPORT_FILE,
        help="Output Markdown report filename.",
    )
    parser.add_argument(
        "--report-json-file",
        default=DEFAULT_REPORT_JSON_FILE,
        help="Output JSON report filename.",
    )
    parser.add_argument(
        "--content-field",
        default="retrieval_text",
        help="Record field used as AutoRAG contents.",
    )
    parser.add_argument(
        "--doc-id-field",
        default="id",
        help="Record field used as AutoRAG doc_id.",
    )
    parser.add_argument(
        "--metadata-fields",
        default=",".join(DEFAULT_METADATA_FIELDS),
        help="Comma-separated metadata fields copied from normalized records.",
    )
    parser.add_argument(
        "--verify-readback",
        action="store_true",
        help="Read the parquet file back after writing and verify row count.",
    )

    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    corpus_path = out_dir / args.corpus_file
    metadata_fields = parse_metadata_fields(args.metadata_fields)
    generated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    records = load_jsonl(input_jsonl)
    rows = build_corpus_rows(
        records=records,
        content_field=args.content_field,
        doc_id_field=args.doc_id_field,
        metadata_fields=metadata_fields,
        generated_at=generated_at,
    )

    validate_rows(rows)
    write_parquet(rows, corpus_path)

    if args.verify_readback:
        dataframe = read_back_parquet(corpus_path)

        if len(dataframe) != len(rows):
            raise ValueError(
                f"Readback row count mismatch: expected {len(rows)}, got {len(dataframe)}"
            )

    report = build_report_data(
        input_jsonl=input_jsonl,
        corpus_path=corpus_path,
        records=records,
        rows=rows,
        content_field=args.content_field,
        doc_id_field=args.doc_id_field,
        metadata_fields=metadata_fields,
    )

    md_path, json_path = write_reports(
        report=report,
        out_dir=out_dir,
        md_name=args.report_file,
        json_name=args.report_json_file,
    )

    print(f"Input JSONL: {input_jsonl}")
    print(f"Rows exported: {len(rows)}")
    print(f"Saved corpus parquet: {corpus_path}")
    print(f"Saved Markdown report: {md_path}")
    print(f"Saved JSON report: {json_path}")


if __name__ == "__main__":
    main()
