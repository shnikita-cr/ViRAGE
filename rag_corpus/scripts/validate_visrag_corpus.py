from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visrag_corpus_utils import (  # noqa: E402
    coverage,
    flatten_retrieval_gt,
    iter_json_records,
    normalize_qa_record,
    normalize_record,
    validate_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalized ViRAGE VisRAG corpora and QA benchmarks.")
    parser.add_argument("--corpus-root", required=True, help="Folder with normalized corpus JSON/JSONL files")
    parser.add_argument("--qa-root", default=None, help="Folder with QA benchmark JSON/JSONL files")
    parser.add_argument("--out-dir", default="rag_corpus/reports", help="Output report directory")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status on validation errors")
    args = parser.parse_args()

    corpus_root = Path(args.corpus_root).resolve()
    qa_root = Path(args.qa_root).resolve() if args.qa_root else None
    out_dir = Path(args.out_dir).resolve()

    rows, row_errors = load_and_validate_corpus(corpus_root)
    qa_rows, qa_errors = load_and_validate_qa(qa_root, {str(row["id"]) for row in rows}) if qa_root else ([], [])

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "corpus_coverage.json", coverage(rows, qa_rows))
    write_json(out_dir / "corpus_validation_errors.json", {"corpus_errors": row_errors, "qa_errors": qa_errors})
    write_json(out_dir / "corpus_manifest.json", build_manifest(corpus_root, rows, qa_rows, row_errors, qa_errors))

    error_count = len(row_errors) + len(qa_errors)
    print(f"Corpus root: {corpus_root}")
    if qa_root:
        print(f"QA root:     {qa_root}")
    print(f"Records:     {len(rows)}")
    print(f"QA records:  {len(qa_rows)}")
    print(f"Errors:      {error_count}")
    print(f"Reports:     {out_dir}")

    if args.strict and error_count:
        raise SystemExit(1)


def load_and_validate_corpus(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not root.exists():
        raise FileNotFoundError(f"Corpus root does not exist: {root}")
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()

    for source_file in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".ndjson"}):
        corpus_name = source_file.stem
        for raw in iter_json_records(source_file, include_control_files=False):
            row = normalize_record(raw, default_corpus=corpus_name)
            row_errors = validate_record(row)
            record_id = str(row.get("id") or "")
            if record_id in seen:
                row_errors.append("duplicate_id")
            if record_id:
                seen.add(record_id)
            if row_errors:
                errors.append({"id": record_id, "file": source_file.as_posix(), "errors": row_errors})
            else:
                rows.append(row)
    return rows, errors



def _iter_qa_files(root: Path):
    """Yield active QA benchmark files.

    The deprecated chart_retrieval_qa.jsonl file used chart type names
    (bar/line/point) as retrieval_gt. Current AutoRAG QA must reference
    real corpus document ids, so the old file is ignored when it is still
    present in a working tree.
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".ndjson"}:
            continue
        if path.name == "chart_retrieval_qa.jsonl":
            continue
        yield path


def load_and_validate_qa(root: Path, doc_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not root.exists():
        raise FileNotFoundError(f"QA root does not exist: {root}")
    qa_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()

    for source_file in _iter_qa_files(root):
        for raw in iter_json_records(source_file, include_control_files=True):
            try:
                row = normalize_qa_record(raw)
            except ValueError as exc:
                errors.append({"file": source_file.as_posix(), "errors": [str(exc)]})
                continue
            row_errors: list[str] = []
            if row["qid"] in seen:
                row_errors.append("duplicate_qid")
            seen.add(row["qid"])
            missing = [doc_id for doc_id in flatten_retrieval_gt(row["retrieval_gt"]) if doc_id not in doc_ids]
            if missing:
                row_errors.append(f"missing_retrieval_gt:{missing}")
            if row_errors:
                errors.append({"qid": row["qid"], "file": source_file.as_posix(), "errors": row_errors})
            else:
                qa_rows.append(row)
    return qa_rows, errors


def build_manifest(
    corpus_root: Path,
    rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    row_errors: list[dict[str, Any]],
    qa_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    files = defaultdict(int)
    for path in sorted(corpus_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
            files[path.name] += 1
    return {
        "schema_version": "1.0",
        "format": "virage-visrag-normalized-corpus-v1",
        "corpus_root": corpus_root.as_posix(),
        "files": sorted(files),
        "records": len(rows),
        "qa_records": len(qa_rows),
        "corpus_errors": len(row_errors),
        "qa_errors": len(qa_errors),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
