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

from scripts.rag_corpus.common.io import project_root, read_jsonl, write_json, write_jsonl
from scripts.rag_corpus.common.schemas import RagRuleRecord, RuntimeRuleDocument

DEFAULT_INPUT = "rag_corpus/processed/all_rules.validated.jsonl"
DEFAULT_OUTPUT = "rag_corpus/runtime/virage_rules.jsonl"
DEFAULT_REPORT = "rag_corpus/runtime/runtime_export_report.json"
DEFAULT_SOURCE_LIMITS = {"chartsquared": 1500}


def export_runtime_rules(
        input_path: Path,
        output_path: Path,
        *,
        source_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    records = [RagRuleRecord.model_validate(raw) for raw in read_jsonl(input_path)]
    docs: list[dict[str, Any]] = []
    skipped_by_source_limit: dict[str, int] = {}
    used_by_source: dict[str, int] = {}
    limits = source_limits if source_limits is not None else dict(DEFAULT_SOURCE_LIMITS)
    for record in records:
        source_dataset = record.source.dataset
        limit = limits.get(source_dataset)
        if limit is not None and used_by_source.get(source_dataset, 0) >= limit:
            skipped_by_source_limit[source_dataset] = skipped_by_source_limit.get(source_dataset, 0) + 1
            continue
        used_by_source[source_dataset] = used_by_source.get(source_dataset, 0) + 1
        source_weight = record.metadata.get("source_weight") if isinstance(record.metadata, dict) else None
        if source_weight is None and source_dataset == "chartsquared":
            source_weight = 0.75
        docs.append(RuntimeRuleDocument(
            doc_id=record.doc_id,
            record_type=record.record_type,
            retrieval_text=record.retrieval_text,
            prompt_text=record.prompt_text,
            metadata={
                "title": record.title,
                "task": record.task,
                "chart_family": record.chart_family,
                "severity": record.severity,
                "avoid": record.avoid,
                "applies_when": record.applies_when,
                "guidance": record.guidance,
                "source_dataset": source_dataset,
                "source_id": record.source.source_id,
                "source_weight": source_weight if source_weight is not None else 1.0,
            },
        ).model_dump())
    write_jsonl(output_path, docs)
    by_type: dict[str, int] = {}
    for doc in docs:
        by_type[doc["record_type"]] = by_type.get(doc["record_type"], 0) + 1
    by_source: dict[str, int] = {}
    for doc in docs:
        source = str((doc.get("metadata") or {}).get("source_dataset") or "unknown")
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "documents": len(docs),
        "by_record_type": by_type,
        "by_source_dataset": by_source,
        "skipped_by_source_limit": skipped_by_source_limit,
        "source_limits": limits,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validated rule records to ViRAGE runtime rule corpus.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--max-chartsquared-docs", type=int, default=1500)
    args = parser.parse_args()
    root = project_root()
    source_limits = {"chartsquared": max(0, int(args.max_chartsquared_docs))}
    report = export_runtime_rules(root / args.input, root / args.output, source_limits=source_limits)
    write_json(root / args.report, report)
    print(report)


if __name__ == "__main__":
    main()
