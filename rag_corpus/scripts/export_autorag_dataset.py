from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visrag_corpus_utils import (  # noqa: E402
    autorag_contents,
    autorag_metadata,
    coverage,
    flatten_retrieval_gt,
    iter_json_records,
    normalize_qa_record,
    normalize_record,
    validate_record,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export normalized ViRAGE VisRAG corpus and QA folders to AutoRAG parquet.")
    parser.add_argument("--corpus-src", required=True, help="Folder with normalized corpus JSON/JSONL files")
    parser.add_argument("--qa-src", required=True, help="Folder with QA benchmark JSON/JSONL files")
    parser.add_argument("--out-dir", required=True, help="Output folder for corpus.parquet and qa.parquet")
    parser.add_argument("--strict", action="store_true", help="Fail on invalid corpus or QA references")
    args = parser.parse_args()

    corpus_src = Path(args.corpus_src).resolve()
    qa_src = Path(args.qa_src).resolve()
    out_dir = Path(args.out_dir).resolve()

    corpus_records, corpus_errors = load_corpus(corpus_src)
    qa_records, qa_errors = load_qa(qa_src, {str(row["id"]) for row in corpus_records})

    if args.strict and (corpus_errors or qa_errors):
        raise ValueError(f"AutoRAG export validation failed: corpus_errors={len(corpus_errors)}, qa_errors={len(qa_errors)}")
    if not corpus_records:
        raise RuntimeError("No valid corpus records found.")
    if not qa_records:
        raise RuntimeError("No valid QA records found.")

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_rows = [to_autorag_corpus_row(row) for row in corpus_records]
    qa_rows = [to_autorag_qa_row(row) for row in qa_records]

    write_jsonl(out_dir / "corpus.preview.jsonl", corpus_rows)
    write_jsonl(out_dir / "qa.preview.jsonl", qa_rows)
    write_report(out_dir / "autorag_export_report.json", corpus_src, qa_src, corpus_records, qa_records, corpus_errors, qa_errors)
    write_parquet(out_dir / "corpus.parquet", corpus_rows)
    write_parquet(out_dir / "qa.parquet", qa_rows)

    print(f"Corpus rows: {len(corpus_rows)} -> {out_dir / 'corpus.parquet'}")
    print(f"QA rows:     {len(qa_rows)} -> {out_dir / 'qa.parquet'}")
    print(f"Report:      {out_dir / 'autorag_export_report.json'}")


def load_corpus(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".ndjson"}):
        default_corpus = path.stem
        for raw in iter_json_records(path, include_control_files=False):
            row = normalize_record(raw, default_corpus=default_corpus)
            row_errors = validate_record(row)
            record_id = str(row.get("id") or "")
            if record_id in seen:
                row_errors.append("duplicate_id")
            if record_id:
                seen.add(record_id)
            if row_errors:
                errors.append({"id": record_id, "file": path.as_posix(), "errors": row_errors})
            else:
                rows.append(row)
    return rows, errors



def _iter_qa_files(root: Path):
    """Yield active QA benchmark files and ignore deprecated local leftovers."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".ndjson"}:
            continue
        if path.name == "chart_retrieval_qa.jsonl":
            continue
        yield path


def load_qa(root: Path, doc_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _iter_qa_files(root):
        for raw in iter_json_records(path, include_control_files=True):
            try:
                row = normalize_qa_record(raw)
            except ValueError as exc:
                errors.append({"file": path.as_posix(), "errors": [str(exc)]})
                continue
            row_errors: list[str] = []
            if row["qid"] in seen:
                row_errors.append("duplicate_qid")
            seen.add(row["qid"])
            missing = [doc_id for doc_id in flatten_retrieval_gt(row["retrieval_gt"]) if doc_id not in doc_ids]
            if missing:
                row_errors.append(f"missing_retrieval_gt:{missing}")
            if row_errors:
                errors.append({"qid": row["qid"], "file": path.as_posix(), "errors": row_errors})
            else:
                rows.append(row)
    return rows, errors


def to_autorag_corpus_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": str(row["id"]),
        "contents": autorag_contents(row),
        "metadata": autorag_metadata(row),
    }


def to_autorag_qa_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": str(row["qid"]),
        "query": str(row["query"]),
        "retrieval_gt": row["retrieval_gt"],
        "generation_gt": row["generation_gt"] if isinstance(row["generation_gt"], list) else [str(row["generation_gt"])],
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("AutoRAG export requires pandas.") from exc
    try:
        pd.DataFrame(rows).to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError("AutoRAG parquet export requires pyarrow or fastparquet. Install: pip install pyarrow") from exc


def write_report(
    path: Path,
    corpus_src: Path,
    qa_src: Path,
    corpus_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    corpus_errors: list[dict[str, Any]],
    qa_errors: list[dict[str, Any]],
) -> None:
    payload = {
        "corpus_src": corpus_src.as_posix(),
        "qa_src": qa_src.as_posix(),
        "corpus_rows": len(corpus_rows),
        "qa_rows": len(qa_rows),
        "corpus_errors": corpus_errors,
        "qa_errors": qa_errors,
        "coverage": coverage(corpus_rows, qa_rows),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
