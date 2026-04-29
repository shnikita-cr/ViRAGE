from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ViRAGE RAG corpus and QA folders to AutoRAG parquet datasets.")
    parser.add_argument("--corpus-src", required=True, help="Folder with prepared corpus JSON/JSONL files")
    parser.add_argument("--qa-src", required=True, help="Folder with QA JSON/JSONL files")
    parser.add_argument("--out-dir", required=True, help="Output folder for corpus.parquet and qa.parquet")
    args = parser.parse_args()

    corpus_rows = [to_autorag_corpus_row(row) for row in load_rows(Path(args.corpus_src))]
    qa_rows = [to_autorag_qa_row(row) for row in load_rows(Path(args.qa_src), include_control_files=True)]
    if not corpus_rows:
        raise RuntimeError("No corpus rows found.")
    if not qa_rows:
        raise RuntimeError("No QA rows found.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(corpus_rows).to_parquet(out / "corpus.parquet", index=False)
    pd.DataFrame(qa_rows).to_parquet(out / "qa.parquet", index=False)
    print(f"Corpus rows: {len(corpus_rows)} -> {out / 'corpus.parquet'}")
    print(f"QA rows:     {len(qa_rows)} -> {out / 'qa.parquet'}")


def load_rows(root: Path, *, include_control_files: bool = False) -> Iterable[dict[str, Any]]:
    ignored = set() if include_control_files else {"manifest.json", "validation_report.json", "lexical_index.json"}
    for path in sorted([*root.glob("*.jsonl"), *root.glob("*.json")]):
        if path.name in ignored:
            continue
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        yield item
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else payload.get("records", []) if isinstance(payload, dict) else []
            for item in records:
                if isinstance(item, dict):
                    yield item


def to_autorag_corpus_row(row: dict[str, Any]) -> dict[str, Any]:
    doc_id = str(row.get("id") or row.get("doc_id") or "").strip()
    if not doc_id:
        raise ValueError(f"Corpus row missing id: {row!r}")
    contents = "\n".join(
        part
        for part in [
            f"instruction: {row.get('instruction') or ''}",
            f"description: {row.get('description') or ''}",
            f"chart_type: {row.get('chart_type') or ''}",
            f"field_roles: {json.dumps(row.get('field_roles') or {}, ensure_ascii=False)}",
            f"spec_template: {json.dumps(row.get('spec_template') or {}, ensure_ascii=False)}",
        ]
        if part.strip()
    )
    metadata = {
        "source": row.get("source"),
        "corpus": row.get("corpus"),
        "chart_type": row.get("chart_type"),
        "field_roles": row.get("field_roles"),
        "quality": row.get("quality"),
        "source_weight": row.get("source_weight"),
        "last_modified_datetime": "1970-01-01T00:00:00",
    }
    return {"doc_id": doc_id, "contents": contents, "metadata": metadata}


def to_autorag_qa_row(row: dict[str, Any]) -> dict[str, Any]:
    qid = str(row.get("qid") or row.get("id") or "").strip()
    query = str(row.get("query") or row.get("question") or "").strip()
    if not qid or not query:
        raise ValueError(f"QA row must contain id/qid and query: {row!r}")
    retrieval_gt = normalize_retrieval_gt(row.get("retrieval_gt"))
    generation_gt = row.get("generation_gt") or row.get("answer") or row.get("expected_answer") or ""
    return {"qid": qid, "query": query, "retrieval_gt": retrieval_gt, "generation_gt": generation_gt}


def normalize_retrieval_gt(value: Any) -> list[list[str]]:
    if isinstance(value, str):
        return [[value]]
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return [list(value)]
        if all(isinstance(item, list) for item in value):
            return [[str(inner) for inner in group] for group in value]
    raise ValueError(f"retrieval_gt must be string, list[str], or list[list[str]], got: {value!r}")


if __name__ == "__main__":
    main()
