from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import argparse
from pathlib import Path
from typing import Any

from scripts.rag_corpus.common.hashing import stable_hash
from scripts.rag_corpus.common.io import project_root, read_jsonl, write_json, write_jsonl
from scripts.rag_corpus.common.text import compact_text

DEFAULT_INPUT = "rag_corpus/processed/all_rules.jsonl"
DEFAULT_OUTPUT = "rag_corpus/processed/all_rules.deduped.jsonl"
DEFAULT_REJECTED = "rag_corpus/processed/rejected_records.jsonl"


def _dedupe_key(record: dict[str, Any]) -> str:
    return stable_hash([
        record.get("record_type"),
        compact_text(record.get("title"), max_chars=120).lower(),
        compact_text(record.get("retrieval_text"), max_chars=300).lower(),
    ], length=16)


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen_doc_ids: set[str] = set()
    seen_content: set[str] = set()
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        doc_id = str(record.get("doc_id") or "")
        content_key = _dedupe_key(record)
        if doc_id in seen_doc_ids:
            rejected.append({"reason": "duplicate_doc_id", "record": record})
            continue
        if content_key in seen_content:
            rejected.append({"reason": "duplicate_content", "record": record})
            continue
        seen_doc_ids.add(doc_id)
        seen_content.add(content_key)
        kept.append(record)
    return kept, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate processed ViRAGE rule records.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--rejected", default=DEFAULT_REJECTED)
    args = parser.parse_args()
    root = project_root()
    kept, rejected = deduplicate_records(read_jsonl(root / args.input))
    write_jsonl(root / args.output, kept)
    if rejected:
        write_jsonl(root / args.rejected, rejected)
    write_json(root / "rag_corpus/processed/deduplication_report.json", {"kept": len(kept), "rejected": len(rejected)})
    print({"kept": len(kept), "rejected": len(rejected)})


if __name__ == "__main__":
    main()
