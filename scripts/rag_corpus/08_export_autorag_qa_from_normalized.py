from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT_JSONL = "rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl"
DEFAULT_CORPUS_PARQUET = "rag_corpus/autorag/vega_lite/corpus.parquet"
DEFAULT_OUT_DIR = "rag_corpus/autorag/vega_lite"
DEFAULT_QA_FILE = "qa.parquet"
DEFAULT_REPORT_FILE = "qa_export_report.md"
DEFAULT_REPORT_JSON_FILE = "qa_export_report.json"

SUPPORTED_QUERY_MODES = {
    "query_field",
    "title",
    "title_query",
    "chart_pattern",
    "all",
}


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


def load_corpus_doc_ids(path: Path, doc_id_column: str) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus parquet file does not exist: {path}")

    dataframe = pd.read_parquet(path)

    if doc_id_column not in dataframe.columns:
        raise ValueError(
            f"Corpus parquet must contain `{doc_id_column}` column: {path}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    doc_ids: set[str] = set()

    for value in dataframe[doc_id_column].tolist():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid doc_id in corpus parquet: {value!r}")

        doc_ids.add(value)

    return doc_ids


def optional_value(record: dict[str, Any], field: str | None) -> Any:
    if not field:
        return None

    return record.get(field)


def optional_str(record: dict[str, Any], field: str | None, default: str = "") -> str:
    value = optional_value(record, field)

    if isinstance(value, str):
        return value.strip()

    if value is None:
        return default

    return str(value).strip()


def required_str(record: dict[str, Any], field: str, context: str) -> str:
    value = record.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: record has invalid `{field}` value: {value!r}")

    return value.strip()


def stringify_mapping_like(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return "none"

        return ", ".join(
            f"{key} {item}"
            for key, item in sorted(value.items())
        )

    if isinstance(value, list):
        if not value:
            return "none"

        return ", ".join(str(item) for item in value)

    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return "none"

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped

        return stringify_mapping_like(parsed)

    if value is None:
        return "none"

    return str(value)


def normalize_generation_gt(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]

    if isinstance(value, list):
        result: list[str] = []

        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
            elif item is not None:
                result.append(str(item).strip())

        if result:
            return result

    return []


def build_generation_gt(
        record: dict[str, Any],
        generation_field: str | None,
        query_field: str,
        chart_pattern_field: str | None,
        mark_type_field: str | None,
        field_roles_field: str | None,
) -> list[str]:
    explicit_generation_gt = normalize_generation_gt(optional_value(record, generation_field))

    if explicit_generation_gt:
        return explicit_generation_gt

    chart_pattern = optional_str(record, chart_pattern_field, default="unknown_pattern")
    mark_type = optional_str(record, mark_type_field, default="unknown_mark")
    field_roles_text = stringify_mapping_like(optional_value(record, field_roles_field))

    if chart_pattern != "unknown_pattern" or mark_type != "unknown_mark" or field_roles_text != "none":
        return [
            (
                f"{chart_pattern} using {mark_type} mark "
                f"with field roles: {field_roles_text}"
            )
        ]

    query = optional_str(record, query_field)

    if query:
        return [query]

    return ["relevant document for the query"]


def build_query(
        record: dict[str, Any],
        query_mode: str,
        query_field: str,
        title_field: str | None,
        chart_pattern_field: str | None,
        mark_type_field: str | None,
        field_roles_field: str | None,
) -> str:
    query_text = required_str(record, query_field, context=f"query_mode={query_mode}")
    title = optional_str(record, title_field)
    chart_pattern = optional_str(record, chart_pattern_field)
    mark_type = optional_str(record, mark_type_field)
    field_roles_text = stringify_mapping_like(optional_value(record, field_roles_field))

    if query_mode == "query_field":
        return query_text

    if query_mode == "title":
        if title:
            return title
        return query_text

    if query_mode == "title_query":
        if title:
            return f"{title}. {query_text}"
        return query_text

    if query_mode == "chart_pattern":
        parts = ["Find a relevant example"]

        if chart_pattern:
            parts.append(f"for {chart_pattern}")

        if mark_type:
            parts.append(f"using {mark_type} mark")

        if field_roles_text != "none":
            parts.append(f"with field roles: {field_roles_text}")

        return " ".join(parts) + "."

    if query_mode == "all":
        parts = []

        if title:
            parts.append(f"Title: {title}.")

        parts.append(f"Query: {query_text}")

        if chart_pattern:
            parts.append(f"Chart pattern: {chart_pattern}.")

        if mark_type:
            parts.append(f"Mark type: {mark_type}.")

        if field_roles_text != "none":
            parts.append(f"Field roles: {field_roles_text}.")

        return " ".join(parts)

    raise ValueError(f"Unsupported query_mode: {query_mode}")


def build_metadata(
        record: dict[str, Any],
        metadata_fields: list[str],
        source_doc_id: str,
        query_mode: str,
        record_index: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_doc_id": source_doc_id,
        "query_mode": query_mode,
        "record_index": record_index,
    }

    for field in metadata_fields:
        if not field:
            continue

        if field not in record:
            continue

        value = record[field]

        if isinstance(value, (dict, list)):
            metadata[field] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif value is None or isinstance(value, (str, int, float, bool)):
            metadata[field] = value
        else:
            metadata[field] = str(value)

    return metadata


def build_qa_rows(
        records: list[dict[str, Any]],
        corpus_doc_ids: set[str],
        query_modes: list[str],
        qid_prefix: str,
        id_field: str,
        query_field: str,
        title_field: str | None,
        generation_field: str | None,
        chart_pattern_field: str | None,
        mark_type_field: str | None,
        field_roles_field: str | None,
        metadata_fields: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_number = 1

    for record_index, record in enumerate(records, start=1):
        doc_id = required_str(record, id_field, context=f"record_index={record_index}")

        if doc_id not in corpus_doc_ids:
            raise ValueError(
                f"Record `{id_field}` does not exist in corpus.parquet doc_id column: {doc_id}"
            )

        generation_gt = build_generation_gt(
            record=record,
            generation_field=generation_field,
            query_field=query_field,
            chart_pattern_field=chart_pattern_field,
            mark_type_field=mark_type_field,
            field_roles_field=field_roles_field,
        )

        for query_mode in query_modes:
            query = build_query(
                record=record,
                query_mode=query_mode,
                query_field=query_field,
                title_field=title_field,
                chart_pattern_field=chart_pattern_field,
                mark_type_field=mark_type_field,
                field_roles_field=field_roles_field,
            )

            if not query.strip():
                raise ValueError(f"Empty query for doc_id={doc_id}, mode={query_mode}")

            rows.append(
                {
                    "qid": f"{qid_prefix}_{row_number:06d}",
                    "query": query,
                    "retrieval_gt": [[doc_id]],
                    "generation_gt": generation_gt,
                    "metadata": build_metadata(
                        record=record,
                        metadata_fields=metadata_fields,
                        source_doc_id=doc_id,
                        query_mode=query_mode,
                        record_index=record_index,
                    ),
                }
            )
            row_number += 1

    return rows


def validate_qa_rows(rows: list[dict[str, Any]], corpus_doc_ids: set[str]) -> None:
    qids = [row["qid"] for row in rows]
    duplicated_qids = [qid for qid, count in Counter(qids).items() if count > 1]

    if duplicated_qids:
        preview = ", ".join(duplicated_qids[:20])
        raise ValueError(f"Duplicate qid values found: {preview}")

    for row in rows:
        qid = row.get("qid")
        query = row.get("query")
        retrieval_gt = row.get("retrieval_gt")
        generation_gt = row.get("generation_gt")

        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"Invalid qid: {row!r}")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Invalid query for qid={qid}: {query!r}")

        if not isinstance(retrieval_gt, list) or not retrieval_gt:
            raise ValueError(f"Invalid retrieval_gt for qid={qid}: {retrieval_gt!r}")

        for group in retrieval_gt:
            if not isinstance(group, list) or not group:
                raise ValueError(f"Invalid retrieval_gt group for qid={qid}: {group!r}")

            for doc_id in group:
                if not isinstance(doc_id, str) or not doc_id.strip():
                    raise ValueError(f"Invalid retrieval_gt doc_id for qid={qid}: {doc_id!r}")

                if doc_id not in corpus_doc_ids:
                    raise ValueError(
                        f"retrieval_gt doc_id does not exist in corpus: {doc_id}"
                    )

        if not isinstance(generation_gt, list) or not generation_gt:
            raise ValueError(f"Invalid generation_gt for qid={qid}: {generation_gt!r}")

        for answer in generation_gt:
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Invalid generation_gt answer for qid={qid}: {answer!r}")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.reset_index(drop=True)

    dataframe.to_parquet(path, index=False)


def verify_readback(path: Path, expected_rows: int) -> None:
    dataframe = pd.read_parquet(path)

    if len(dataframe) != expected_rows:
        raise ValueError(
            f"Readback row count mismatch: expected {expected_rows}, got {len(dataframe)}"
        )

    required_columns = {"qid", "query", "retrieval_gt", "generation_gt"}

    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Readback parquet is missing columns: {sorted(missing)}")


def count_record_field(records: list[dict[str, Any]], field: str | None) -> dict[str, int]:
    if not field:
        return {}

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


def count_row_metadata(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()

    for row in rows:
        metadata = row.get("metadata", {})

        if not isinstance(metadata, dict):
            counter["[invalid_metadata]"] += 1
            continue

        value = metadata.get(field)

        if isinstance(value, str):
            counter[value] += 1
        elif value is None:
            counter["[null]"] += 1
        else:
            counter[str(value)] += 1

    return dict(counter.most_common())


def build_report_data(
        input_jsonl: Path,
        corpus_parquet: Path,
        qa_path: Path,
        records: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        query_modes: list[str],
        id_field: str,
        query_field: str,
        title_field: str | None,
        generation_field: str | None,
        chart_pattern_field: str | None,
        mark_type_field: str | None,
        field_roles_field: str | None,
        metadata_fields: list[str],
) -> dict[str, Any]:
    query_lengths = [len(row["query"]) for row in rows]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": input_jsonl.as_posix(),
        "corpus_parquet": corpus_parquet.as_posix(),
        "qa_path": qa_path.as_posix(),
        "normalized_record_count": len(records),
        "qa_row_count": len(rows),
        "query_modes": query_modes,
        "id_field": id_field,
        "query_field": query_field,
        "title_field": title_field,
        "generation_field": generation_field,
        "chart_pattern_field": chart_pattern_field,
        "mark_type_field": mark_type_field,
        "field_roles_field": field_roles_field,
        "metadata_fields": metadata_fields,
        "min_query_length": min(query_lengths) if query_lengths else 0,
        "max_query_length": max(query_lengths) if query_lengths else 0,
        "avg_query_length": round(sum(query_lengths) / len(query_lengths), 2)
        if query_lengths
        else 0,
        "chart_pattern_distribution_from_records": count_record_field(records, chart_pattern_field),
        "mark_type_distribution_from_records": count_record_field(records, mark_type_field),
        "query_mode_distribution": count_row_metadata(rows, "query_mode"),
        "sample_rows": [
            {
                "qid": row["qid"],
                "query": row["query"],
                "retrieval_gt": row["retrieval_gt"],
                "generation_gt": row["generation_gt"],
                "metadata": row["metadata"],
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

    lines.append("# AutoRAG QA export report")
    lines.append("")
    lines.append(f"Generated at: `{report['generated_at']}`")
    lines.append(f"Input JSONL: `{report['input_jsonl']}`")
    lines.append(f"Corpus parquet: `{report['corpus_parquet']}`")
    lines.append(f"QA parquet: `{report['qa_path']}`")
    lines.append(f"Normalized records: **{report['normalized_record_count']}**")
    lines.append(f"QA rows: **{report['qa_row_count']}**")
    lines.append(f"Query modes: `{', '.join(report['query_modes'])}`")
    lines.append("")
    lines.append("## Field mapping")
    lines.append("")
    lines.append(f"- id_field: `{report['id_field']}`")
    lines.append(f"- query_field: `{report['query_field']}`")
    lines.append(f"- title_field: `{report['title_field']}`")
    lines.append(f"- generation_field: `{report['generation_field']}`")
    lines.append(f"- chart_pattern_field: `{report['chart_pattern_field']}`")
    lines.append(f"- mark_type_field: `{report['mark_type_field']}`")
    lines.append(f"- field_roles_field: `{report['field_roles_field']}`")
    lines.append("")
    lines.append("## Query length")
    lines.append("")
    lines.append(f"- Min query length: `{report['min_query_length']}`")
    lines.append(f"- Max query length: `{report['max_query_length']}`")
    lines.append(f"- Avg query length: `{report['avg_query_length']}`")
    lines.append("")

    lines.extend(
        markdown_counter(
            "Chart pattern distribution from normalized records",
            report["chart_pattern_distribution_from_records"],
        )
    )
    lines.extend(
        markdown_counter(
            "Mark type distribution from normalized records",
            report["mark_type_distribution_from_records"],
        )
    )
    lines.extend(
        markdown_counter(
            "Query mode distribution",
            report["query_mode_distribution"],
        )
    )

    lines.append("## Sample rows")
    lines.append("")
    lines.append("| qid | query | retrieval_gt | generation_gt |")
    lines.append("|---|---|---|---|")

    for row in report["sample_rows"]:
        query = str(row["query"]).replace("|", "\\|").replace("\n", " ")
        retrieval_gt = json.dumps(row["retrieval_gt"], ensure_ascii=False)
        generation_gt = json.dumps(row["generation_gt"], ensure_ascii=False)
        lines.append(
            f"| `{row['qid']}` | {query} | `{retrieval_gt}` | `{generation_gt}` |"
        )

    lines.append("")
    return "\n".join(lines)


def write_reports(
        report: dict[str, Any],
        out_dir: Path,
        report_file: str,
        report_json_file: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / report_file
    json_path = out_dir / report_json_file

    md_path.write_text(build_markdown_report(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return md_path, json_path


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_optional_field(value: str) -> str | None:
    stripped = value.strip()

    if not stripped or stripped.lower() in {"none", "null", "-"}:
        return None

    return stripped


def parse_query_modes(value: str) -> list[str]:
    modes = parse_csv_arg(value)

    if not modes:
        return ["query_field"]

    unsupported = [mode for mode in modes if mode not in SUPPORTED_QUERY_MODES]

    if unsupported:
        raise ValueError(
            f"Unsupported query modes: {unsupported}. "
            f"Supported values: {sorted(SUPPORTED_QUERY_MODES)}"
        )

    return modes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export reusable AutoRAG qa.parquet from normalized JSONL records."
    )

    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Path to normalized JSONL file.",
    )
    parser.add_argument(
        "--corpus-parquet",
        default=DEFAULT_CORPUS_PARQUET,
        help="Path to AutoRAG corpus.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for qa.parquet and reports.",
    )
    parser.add_argument(
        "--qa-file",
        default=DEFAULT_QA_FILE,
        help="Output qa parquet filename.",
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
        "--query-modes",
        default="query_field",
        help=(
            "Comma-separated query generation modes. "
            "Supported: query_field,title,title_query,chart_pattern,all"
        ),
    )
    parser.add_argument(
        "--qid-prefix",
        default="vega_lite_qa",
        help="Prefix for generated qid values.",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Normalized JSONL field used as source doc_id.",
    )
    parser.add_argument(
        "--corpus-doc-id-column",
        default="doc_id",
        help="Corpus parquet column that contains doc ids.",
    )
    parser.add_argument(
        "--query-field",
        default="instruction",
        help="Normalized JSONL field used as primary query text.",
    )
    parser.add_argument(
        "--title-field",
        default="title",
        help="Optional normalized JSONL title field. Use 'none' to disable.",
    )
    parser.add_argument(
        "--generation-field",
        default="none",
        help=(
            "Optional field that already contains generation_gt. "
            "Can be string or list[str]. Use 'none' to auto-generate."
        ),
    )
    parser.add_argument(
        "--chart-pattern-field",
        default="chart_pattern",
        help="Optional chart pattern field. Use 'none' to disable.",
    )
    parser.add_argument(
        "--mark-type-field",
        default="mark_type",
        help="Optional mark type field. Use 'none' to disable.",
    )
    parser.add_argument(
        "--field-roles-field",
        default="field_roles",
        help="Optional field roles field. Use 'none' to disable.",
    )
    parser.add_argument(
        "--metadata-fields",
        default="source,title,chart_pattern,mark_type,field_roles,source_path,corpus_type",
        help="Comma-separated normalized fields copied into QA metadata.",
    )
    parser.add_argument(
        "--verify-readback",
        action="store_true",
        help="Read the parquet file back after writing and verify columns/row count.",
    )

    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    corpus_parquet = Path(args.corpus_parquet)
    out_dir = Path(args.out_dir)
    qa_path = out_dir / args.qa_file

    query_modes = parse_query_modes(args.query_modes)
    metadata_fields = parse_csv_arg(args.metadata_fields)

    title_field = parse_optional_field(args.title_field)
    generation_field = parse_optional_field(args.generation_field)
    chart_pattern_field = parse_optional_field(args.chart_pattern_field)
    mark_type_field = parse_optional_field(args.mark_type_field)
    field_roles_field = parse_optional_field(args.field_roles_field)

    records = load_jsonl(input_jsonl)
    corpus_doc_ids = load_corpus_doc_ids(
        path=corpus_parquet,
        doc_id_column=args.corpus_doc_id_column,
    )

    rows = build_qa_rows(
        records=records,
        corpus_doc_ids=corpus_doc_ids,
        query_modes=query_modes,
        qid_prefix=args.qid_prefix,
        id_field=args.id_field,
        query_field=args.query_field,
        title_field=title_field,
        generation_field=generation_field,
        chart_pattern_field=chart_pattern_field,
        mark_type_field=mark_type_field,
        field_roles_field=field_roles_field,
        metadata_fields=metadata_fields,
    )

    validate_qa_rows(rows, corpus_doc_ids)
    write_parquet(rows, qa_path)

    if args.verify_readback:
        verify_readback(qa_path, expected_rows=len(rows))

    report = build_report_data(
        input_jsonl=input_jsonl,
        corpus_parquet=corpus_parquet,
        qa_path=qa_path,
        records=records,
        rows=rows,
        query_modes=query_modes,
        id_field=args.id_field,
        query_field=args.query_field,
        title_field=title_field,
        generation_field=generation_field,
        chart_pattern_field=chart_pattern_field,
        mark_type_field=mark_type_field,
        field_roles_field=field_roles_field,
        metadata_fields=metadata_fields,
    )

    md_path, json_path = write_reports(
        report=report,
        out_dir=out_dir,
        report_file=args.report_file,
        report_json_file=args.report_json_file,
    )

    print(f"Input JSONL: {input_jsonl}")
    print(f"Corpus parquet: {corpus_parquet}")
    print(f"QA rows exported: {len(rows)}")
    print(f"Saved QA parquet: {qa_path}")
    print(f"Saved Markdown report: {md_path}")
    print(f"Saved JSON report: {json_path}")


if __name__ == "__main__":
    main()
