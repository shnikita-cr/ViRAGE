from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visrag_corpus_utils import (  # noqa: E402
    coverage,
    iter_json_records,
    normalize_record,
    validate_record,
    write_jsonl,
)

QA_LIKE_FILES = {"autorag_qa_examples.jsonl", "chart_retrieval_qa.jsonl"}
RUNTIME_CORPUS_TYPES = {"example", "rule", "design_constraint", "success_case", "failure_case", "utterance", "ambiguity_case"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize seed JSON/JSONL corpora into the current ViRAGE VisRAG schema.")
    parser.add_argument("--src", default="rag_corpus/examples", help="Source folder with seed corpus files")
    parser.add_argument("--out", default="rag_corpus/normalized/jsonl", help="Output folder for normalized source JSONL files")
    parser.add_argument("--runtime-out", default="rag_corpus/data/visrag_runtime.jsonl", help="Runtime-ready merged corpus JSONL")
    parser.add_argument("--reports-out", default="rag_corpus/reports", help="Output folder for normalization report")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    runtime_out = Path(args.runtime_out).resolve()
    reports_out = Path(args.reports_out).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source corpus folder does not exist: {src}")

    normalized_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    runtime_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for path in sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".ndjson"}):
        if path.name in QA_LIKE_FILES:
            continue
        target_name = path.name if path.name.endswith(".jsonl") else f"{path.stem}.jsonl"
        for raw in iter_json_records(path, include_control_files=True):
            row = normalize_record(raw, default_corpus=path.stem)
            row_errors = validate_record(row)
            if row_errors:
                errors.append({"id": row.get("id"), "file": path.as_posix(), "errors": row_errors})
                continue
            normalized_by_file[target_name].append(row)
            if is_runtime_ready(row):
                runtime_rows.append(to_runtime_row(row))

    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in sorted(normalized_by_file.items()):
        write_jsonl(out / filename, rows)
    write_jsonl(runtime_out, runtime_rows)

    reports_out.mkdir(parents=True, exist_ok=True)
    report = {
        "source": src.as_posix(),
        "normalized_out": out.as_posix(),
        "runtime_out": runtime_out.as_posix(),
        "source_files": {filename: len(rows) for filename, rows in sorted(normalized_by_file.items())},
        "runtime_records": len(runtime_rows),
        "errors": errors,
        "coverage": coverage([row for rows in normalized_by_file.values() for row in rows]),
        "runtime_coverage": coverage(runtime_rows),
    }
    (reports_out / "normalization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"Normalized files: {len(normalized_by_file)} -> {out}")
    print(f"Runtime records:  {len(runtime_rows)} -> {runtime_out}")
    print(f"Errors:           {len(errors)}")
    for key, value in sorted(Counter(row.get("mark_type") or "" for row in runtime_rows).items()):
        print(f"  - {key}: {value}")


def is_runtime_ready(row: dict[str, Any]) -> bool:
    return (
        row.get("corpus_type") in RUNTIME_CORPUS_TYPES
        and bool(row.get("instruction"))
        and bool(row.get("mark_type"))
        and isinstance(row.get("field_roles"), dict)
        and bool(row.get("field_roles"))
        and isinstance(row.get("spec_template"), dict)
        and bool(row.get("spec_template"))
    )


def to_runtime_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "schema_version": row.get("schema_version"),
            "corpus_type": row.get("corpus_type"),
            "quality_tier": row.get("quality_tier"),
            "source_weight": row.get("source_weight"),
            "chart_pattern": row.get("chart_pattern"),
            "task_type": row.get("task_type"),
            "source_url": row.get("source_url"),
        }
    )
    return {
        "id": row["id"],
        "source": row["source"],
        "corpus": row.get("corpus_type") or row.get("source"),
        "instruction": row["instruction"],
        "chart_type": row["mark_type"],
        "description": row.get("description") or row.get("instruction"),
        "keywords": row.get("keywords") or row.get("tags") or [],
        "field_roles": row.get("field_roles") or {},
        "transform_types": row.get("transform_types") or row.get("transforms") or [],
        "spec_template": row.get("spec_template") or {},
        "metadata": metadata,
    }


if __name__ == "__main__":
    main()
