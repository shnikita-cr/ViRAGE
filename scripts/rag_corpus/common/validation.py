from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


from collections import Counter
from typing import Any

from pydantic import ValidationError

from scripts.rag_corpus.common.schemas import RagRuleRecord


def validate_rule_records(raw_records: list[dict[str, Any]]) -> tuple[list[RagRuleRecord], list[dict[str, Any]]]:
    valid: list[RagRuleRecord] = []
    rejected: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for idx, raw in enumerate(raw_records):
        try:
            record = RagRuleRecord.model_validate(raw)
        except ValidationError as exc:
            rejected.append({"index": idx, "reason": exc.errors(), "record": raw})
            continue
        seen[record.doc_id] += 1
        valid.append(record)
    duplicate_ids = {doc_id for doc_id, count in seen.items() if count > 1}
    if duplicate_ids:
        filtered: list[RagRuleRecord] = []
        used: set[str] = set()
        for record in valid:
            if record.doc_id in duplicate_ids and record.doc_id in used:
                rejected.append({"reason": "duplicate_doc_id", "doc_id": record.doc_id, "record": record.model_dump()})
                continue
            used.add(record.doc_id)
            filtered.append(record)
        valid = filtered
    return valid, rejected


def summarize_rule_records(records: list[RagRuleRecord]) -> dict[str, Any]:
    by_type = Counter(record.record_type for record in records)
    by_source = Counter(record.source.dataset for record in records)
    return {
        "total": len(records),
        "by_record_type": dict(sorted(by_type.items())),
        "by_source_dataset": dict(sorted(by_source.items())),
        "avg_prompt_text_chars": round(sum(len(r.prompt_text) for r in records) / len(records), 2) if records else 0,
        "avg_retrieval_text_chars": round(sum(len(r.retrieval_text) for r in records) / len(records), 2) if records else 0,
    }
