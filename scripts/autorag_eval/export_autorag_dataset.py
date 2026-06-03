from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

_CURRENT = Path(__file__).resolve()
ROOT = next((parent for parent in _CURRENT.parents if (parent / "src").exists() and (parent / "scripts").exists()), Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rag_corpus.common.io import ensure_dir, read_jsonl, write_json, write_text

DEFAULT_CORPUS = "rag_corpus/runtime/guidance_chunks.jsonl"
DEFAULT_QUERIES = "rag_corpus/autorag/qa/retrieval_queries.jsonl"
DEFAULT_OUTPUT_DIR = "rag_corpus/autorag/datasets/runtime"


class AutoRAGDatasetExportError(RuntimeError):
    pass


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def _record_id(record: dict[str, Any]) -> str:
    value = record.get("doc_id") or record.get("chunk_id") or record.get("id")
    return str(value or "").strip()


def _record_type(record: dict[str, Any]) -> str:
    return str(record.get("record_type") or record.get("source_kind") or "").strip()


def _source_id(record: dict[str, Any]) -> str:
    return str(record.get("source_id") or _metadata(record).get("source_id") or "").strip()


def _source_name(record: dict[str, Any]) -> str:
    return str(record.get("source_name") or _metadata(record).get("source_name") or "").strip()


def _field_value(record: dict[str, Any], field: str) -> Any:
    if field == "doc_id":
        return _record_id(record)
    if field == "record_type":
        return _record_type(record)
    if field.startswith("metadata."):
        return _metadata(record).get(field.split(".", 1)[1])
    return record.get(field)


def _contains_any(haystack: str, needles: list[str]) -> bool:
    if not needles:
        return True
    text = haystack.lower()
    return any(needle.lower() in text for needle in needles if needle)


def _record_contents(record: dict[str, Any]) -> str:
    metadata = _metadata(record)
    guidance = _string_list(metadata.get("guidance"))
    avoid = _string_list(metadata.get("avoid"))
    applies_when = _string_list(metadata.get("applies_when"))
    title = record.get("title") or metadata.get("title") or ""
    text = record.get("text") or record.get("retrieval_text") or record.get("prompt_text") or ""
    parts: list[str] = [
        f"Title: {title}",
        f"Source: {_source_id(record)} {_source_name(record)}",
        f"Record type: {_record_type(record)}",
        f"Chart family: {metadata.get('chart_family') or ''}",
        f"Task: {metadata.get('task') or ''}",
        f"Applies when: {'; '.join(applies_when)}",
        f"Guidance: {'; '.join(guidance)}",
        f"Avoid: {'; '.join(avoid)}",
        str(text),
    ]
    return "\n".join(part for part in parts if str(part).strip() and str(part).split(":", 1)[-1].strip())


def _corpus_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        doc_id = _record_id(record)
        if not doc_id:
            raise AutoRAGDatasetExportError(f"Corpus record without doc_id/chunk_id/id: {record}")
        if doc_id in seen:
            raise AutoRAGDatasetExportError(f"Duplicate corpus record id: {doc_id}")
        seen.add(doc_id)
        metadata = dict(_metadata(record))
        metadata.setdefault("record_type", _record_type(record))
        metadata.setdefault("source_id", _source_id(record))
        metadata.setdefault("source_name", _source_name(record))
        metadata.setdefault("source_path", record.get("source_path"))
        metadata.setdefault("title", record.get("title"))
        rows.append({
            "doc_id": doc_id,
            "contents": _record_contents(record),
            "metadata": metadata,
        })
    return rows


def _matches_filter(record: dict[str, Any], filter_spec: dict[str, Any]) -> bool:
    for field, expected in filter_spec.items():
        if field in {"text_contains", "contents_contains"}:
            if not _contains_any(_record_contents(record), _string_list(expected)):
                return False
            continue
        if field == "metadata_contains":
            blob = json.dumps(_metadata(record), ensure_ascii=False)
            if not _contains_any(blob, _string_list(expected)):
                return False
            continue
        actual = _field_value(record, field)
        expected_values = _string_list(expected)
        if not expected_values:
            continue
        actual_values = _string_list(actual)
        if not actual_values:
            return False
        actual_lower = {value.lower() for value in actual_values}
        if not any(value.lower() in actual_lower for value in expected_values):
            return False
    return True


def _resolve_retrieval_gt(
    *,
    query: dict[str, Any],
    records: list[dict[str, Any]],
    max_gt_per_query: int,
) -> list[str]:
    by_id = {_record_id(record): record for record in records if _record_id(record)}
    ids: list[str] = []
    for doc_id in _string_list(query.get("relevant_doc_ids")):
        if doc_id in by_id:
            ids.append(doc_id)
    filters = query.get("relevant_filters") or []
    if isinstance(filters, dict):
        filters = [filters]
    if not isinstance(filters, list):
        filters = []
    for filter_spec in filters:
        if not isinstance(filter_spec, dict):
            continue
        for record in records:
            doc_id = _record_id(record)
            if not doc_id or doc_id in ids:
                continue
            if _matches_filter(record, filter_spec):
                ids.append(doc_id)
    if max_gt_per_query > 0:
        ids = ids[:max_gt_per_query]
    return ids


def _qa_rows(
    *,
    queries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    max_gt_per_query: int,
    allow_empty_gt: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    empty_qids: list[str] = []
    for index, query in enumerate(queries, start=1):
        qid = str(query.get("qid") or f"q_{index:04d}").strip()
        text = str(query.get("query") or "").strip()
        if not text:
            raise AutoRAGDatasetExportError(f"Query {qid} has empty query text.")
        retrieval_gt = _resolve_retrieval_gt(query=query, records=records, max_gt_per_query=max_gt_per_query)
        if not retrieval_gt:
            empty_qids.append(qid)
            if not allow_empty_gt:
                continue
        rows.append({
            "qid": qid,
            "query": text,
            "generation_gt": str(query.get("generation_gt") or text),
            "retrieval_gt": retrieval_gt,
            "metadata": {
                "expected_topics": _string_list(query.get("expected_topics")),
                "query_type": query.get("query_type"),
                "source_query": qid,
            },
        })
    return rows, empty_qids


def export_autorag_dataset(
    *,
    corpus_path: Path,
    queries_path: Path,
    output_dir: Path,
    max_gt_per_query: int = 20,
    allow_empty_gt: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)
    if not records:
        raise AutoRAGDatasetExportError(f"Corpus is empty or missing: {corpus_path}")
    if not queries:
        raise AutoRAGDatasetExportError(f"Queries are empty or missing: {queries_path}")

    corpus_rows = _corpus_rows(records)
    qa_rows, empty_qids = _qa_rows(
        queries=queries,
        records=records,
        max_gt_per_query=max_gt_per_query,
        allow_empty_gt=allow_empty_gt,
    )
    if empty_qids and not allow_empty_gt:
        raise AutoRAGDatasetExportError(
            "Some queries have empty retrieval_gt after resolving labels. "
            f"Fix rag_corpus/autorag/qa/retrieval_queries.jsonl. Empty qids: {', '.join(empty_qids[:20])}"
        )
    if not qa_rows:
        raise AutoRAGDatasetExportError("No QA rows were exported.")

    ensure_dir(output_dir)
    corpus_output = output_dir / "corpus.parquet"
    qa_output = output_dir / "qa.parquet"
    pd.DataFrame(corpus_rows).to_parquet(corpus_output, index=False)
    pd.DataFrame(qa_rows).to_parquet(qa_output, index=False)

    source_counts: dict[str, int] = {}
    for record in records:
        source_id = _source_id(record) or "unknown"
        source_counts[source_id] = source_counts.get(source_id, 0) + 1

    report = {
        "corpus_input": str(corpus_path),
        "queries_input": str(queries_path),
        "output_dir": str(output_dir),
        "corpus": {"records": len(corpus_rows), "path": str(corpus_output), "by_source": source_counts},
        "qa": {"records": len(qa_rows), "path": str(qa_output), "empty_qids": empty_qids},
        "max_gt_per_query": max_gt_per_query,
        "allow_empty_gt": allow_empty_gt,
    }
    write_json(output_dir / "dataset_manifest.json", report)
    write_text(
        output_dir / "dataset_manifest.md",
        "\n".join([
            "# ViRAGE AutoRAG dataset export",
            "",
            f"Corpus input: `{corpus_path}`",
            f"Queries input: `{queries_path}`",
            f"Corpus records: **{len(corpus_rows)}**",
            f"QA records: **{len(qa_rows)}**",
            f"Corpus parquet: `{corpus_output}`",
            f"QA parquet: `{qa_output}`",
            "",
        ]),
    )
    return report


def _resolve_optional_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ViRAGE runtime guidance corpus and labelled retrieval queries to AutoRAG parquet files.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-gt-per-query", type=int, default=20)
    parser.add_argument("--allow-empty-gt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_autorag_dataset(
        corpus_path=ROOT / args.corpus,
        queries_path=ROOT / args.queries,
        output_dir=ROOT / args.output_dir,
        max_gt_per_query=args.max_gt_per_query,
        allow_empty_gt=args.allow_empty_gt,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
