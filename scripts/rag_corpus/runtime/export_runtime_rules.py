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


def export_runtime_rules(input_path: Path, output_path: Path) -> dict[str, Any]:
    records = [RagRuleRecord.model_validate(raw) for raw in read_jsonl(input_path)]
    docs: list[dict[str, Any]] = []
    for record in records:
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
                "source_dataset": record.source.dataset,
                "source_id": record.source.source_id,
            },
        ).model_dump())
    write_jsonl(output_path, docs)
    by_type: dict[str, int] = {}
    for doc in docs:
        by_type[doc["record_type"]] = by_type.get(doc["record_type"], 0) + 1
    return {"documents": len(docs), "by_record_type": by_type, "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validated rule records to ViRAGE runtime rule corpus.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()
    root = project_root()
    report = export_runtime_rules(root / args.input, root / args.output)
    write_json(root / args.report, report)
    print(report)


if __name__ == "__main__":
    main()
