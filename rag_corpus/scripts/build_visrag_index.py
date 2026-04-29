from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_CHART_TYPES = {"bar", "line", "area", "point", "circle", "square", "tick", "histogram", "boxplot", "rect", "rule", "text"}
SUPPORTED_ROLES = {"quantitative", "temporal", "nominal", "ordinal", "boolean"}
GENERATED_FILES = {"manifest.json", "validation_report.json", "lexical_index.json"}
TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")

CHART_ALIASES = {
    "scatter": "point",
    "scatterplot": "point",
    "scatter_plot": "point",
    "bar_chart": "bar",
    "line_chart": "line",
    "area_chart": "area",
    "hist": "histogram",
    "box": "boxplot",
    "box_plot": "boxplot",
    "heatmap": "rect",
    "heat_map": "rect",
    "matshow": "rect",
    "imshow": "rect",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a prepared ViRAGE VisRAG corpus and build offline metadata/index files. "
            "The runtime RAG layer only reads prepared corpus files from rag_corpus/data."
        )
    )
    parser.add_argument(
        "--corpus-root",
        default="rag_corpus/data",
        help="Prepared corpus directory. Default: rag_corpus/data",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Directory for manifest, validation report and lexical index. Default: --corpus-root",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any invalid corpus row is found")
    args = parser.parse_args()

    corpus_root = Path(args.corpus_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else corpus_root

    if not corpus_root.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_root}")

    rows = list(load_corpus_rows(corpus_root))
    valid_rows, errors = validate_rows(rows)
    if args.strict and errors:
        raise ValueError(f"Corpus validation failed with {len(errors)} errors. See validation_report.json for details.")

    out_root.mkdir(parents=True, exist_ok=True)
    write_validation_report(out_root / "validation_report.json", corpus_root, valid_rows, errors)
    write_manifest(out_root / "manifest.json", valid_rows, errors)
    write_lexical_index(out_root / "lexical_index.json", valid_rows)

    print(f"Corpus root: {corpus_root}")
    print(f"Output root: {out_root}")
    print(f"Rows:        {len(rows)}")
    print(f"Valid:       {len(valid_rows)}")
    print(f"Errors:      {len(errors)}")
    print("Files written:")
    print(f"  - {out_root / 'manifest.json'}")
    print(f"  - {out_root / 'validation_report.json'}")
    print(f"  - {out_root / 'lexical_index.json'}")


def load_corpus_rows(root: Path) -> Iterable[dict[str, Any]]:
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name not in GENERATED_FILES and path.suffix.lower() in {".jsonl", ".ndjson", ".json"}
    )
    for path in files:
        for index, row in enumerate(read_corpus_file(path), start=1):
            item = dict(row)
            item["_file"] = path.name
            item["_line"] = index
            yield item


def read_corpus_file(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            yield payload
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"JSON list in {path} must contain objects")
            yield item
        return
    if isinstance(payload, dict):
        for key in ("examples", "records", "items", "data", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        raise ValueError(f"JSON key {key!r} in {path} must contain objects")
                    yield item
                return
        yield payload
        return
    raise ValueError(f"Unsupported JSON root in {path}")


def validate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in rows:
        row_errors = validate_row(row)
        example_id = str(row.get("id") or row.get("example_id") or "")
        if example_id in seen_ids:
            row_errors.append("duplicate_id")
        if example_id:
            seen_ids.add(example_id)

        if row_errors:
            errors.append(
                {
                    "file": row.get("_file"),
                    "line": row.get("_line"),
                    "id": example_id,
                    "errors": row_errors,
                }
            )
        else:
            valid_rows.append(normalize_runtime_row(row))
    return valid_rows, errors


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    instruction = row.get("instruction") or row.get("query") or row.get("utterance") or row.get("description")
    if not isinstance(instruction, str) or not instruction.strip():
        errors.append("missing_instruction")

    chart_type = canonicalize_chart_type(row.get("chart_type") or row.get("mark"))
    if chart_type not in SUPPORTED_CHART_TYPES:
        errors.append("unsupported_chart_type")

    field_roles = row.get("field_roles") or row.get("encoding_roles") or {}
    if not isinstance(field_roles, dict):
        errors.append("field_roles_not_dict")
    else:
        for channel, role in field_roles.items():
            if not isinstance(channel, str) or not channel.strip():
                errors.append("invalid_field_role_channel")
            if normalize_role(str(role)) not in SUPPORTED_ROLES:
                errors.append("invalid_field_role_value")

    spec_template = row.get("spec_template") or row.get("spec")
    if not isinstance(spec_template, dict):
        errors.append("spec_template_not_dict")

    return sorted(set(errors))


def normalize_runtime_row(row: dict[str, Any]) -> dict[str, Any]:
    field_roles = {
        str(channel): normalize_role(str(role))
        for channel, role in (row.get("field_roles") or row.get("encoding_roles") or {}).items()
    }
    return {
        "id": str(row.get("id") or row.get("example_id")),
        "source": str(row.get("source") or "local"),
        "corpus": str(row.get("corpus") or row.get("_file") or "local"),
        "instruction": str(
            row.get("instruction") or row.get("query") or row.get("utterance") or row.get("description")),
        "chart_type": canonicalize_chart_type(row.get("chart_type") or row.get("mark")),
        "description": row.get("description") if isinstance(row.get("description"), str) else None,
        "keywords": normalize_keywords(row.get("keywords") or row.get("tags")),
        "field_roles": field_roles,
        "transform_types": [str(item) for item in row.get("transform_types", [])] if isinstance(
            row.get("transform_types", []), list) else [],
        "spec_template": row.get("spec_template") or row.get("spec"),
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        "_file": row.get("_file"),
        "_line": row.get("_line"),
    }


def canonicalize_chart_type(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("type")
    text = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower()).strip("_")
    if text in SUPPORTED_CHART_TYPES:
        return text
    return CHART_ALIASES.get(text, "")


def normalize_role(value: str) -> str:
    text = value.strip().lower()
    if text in {"quantitative", "numeric", "number", "measure", "int", "float", "int64", "float64"}:
        return "quantitative"
    if text in {"temporal", "datetime", "date", "time", "year", "datetime64", "datetime64[ns]"}:
        return "temporal"
    if text in {"nominal", "categorical", "category", "object", "string", "str"}:
        return "nominal"
    if text in {"ordinal", "ordered"}:
        return "ordinal"
    if text in {"bool", "boolean"}:
        return "boolean"
    return ""


def normalize_keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        values = [str(item).strip().lower() for item in value if str(item).strip()]
    elif isinstance(value, str):
        values = [part.strip().lower() for part in re.split(r"[;,|]", value) if part.strip()]
    else:
        values = []
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def tokenize_row(row: dict[str, Any]) -> list[str]:
    text_parts = [
        row.get("instruction") or "",
        row.get("description") or "",
        row.get("chart_type") or "",
        " ".join(row.get("keywords") or []),
        " ".join(row.get("field_roles", {}).values()),
        " ".join(row.get("transform_types") or []),
    ]
    return [token.lower() for token in TOKEN_RE.findall(" ".join(text_parts))]


def write_manifest(path: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    chart_counts = Counter(row["chart_type"] for row in rows)
    payload = {
        "version": "1",
        "format": "virage-visrag-prepared-corpus-v1",
        "files": sorted({str(row.get("_file")) for row in rows if row.get("_file")}),
        "records": len(rows),
        "invalid_records": len(errors),
        "chart_type_counts": dict(sorted(chart_counts.items())),
        "index_files": ["lexical_index.json"],
        "validation_report": "validation_report.json",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_validation_report(path: Path, root: Path, rows: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    payload = {
        "corpus_root": root.as_posix(),
        "valid_records": len(rows),
        "invalid_records": len(errors),
        "error_counts": dict(sorted(Counter(error for item in errors for error in item["errors"]).items())),
        "errors": errors,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_lexical_index(path: Path, rows: list[dict[str, Any]]) -> None:
    documents: list[dict[str, Any]] = []
    inverted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    document_frequency: Counter[str] = Counter()

    for row in rows:
        tokens = tokenize_row(row)
        counts = Counter(tokens)
        unique_tokens = set(counts)
        for token in unique_tokens:
            document_frequency[token] += 1
        doc = {
            "id": row["id"],
            "chart_type": row["chart_type"],
            "instruction": row["instruction"],
            "token_count": len(tokens),
            "tokens": dict(sorted(counts.items())),
        }
        documents.append(doc)
        for token, count in counts.items():
            inverted[token].append({"id": row["id"], "tf": count})

    doc_count = len(documents)
    idf = {token: round(math.log((1 + doc_count) / (1 + df)) + 1.0, 6) for token, df in document_frequency.items()}
    payload = {
        "version": "1",
        "backend": "lexical-v1",
        "documents": documents,
        "document_frequency": dict(sorted(document_frequency.items())),
        "idf": dict(sorted(idf.items())),
        "inverted_index": {token: postings for token, postings in sorted(inverted.items())},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
