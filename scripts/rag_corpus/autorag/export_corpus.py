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

import pandas as pd

from scripts.rag_corpus.common.io import ensure_dir, project_root, read_jsonl, write_json, write_text
from scripts.rag_corpus.common.schemas import RagRuleRecord

DEFAULT_INPUT = "rag_corpus/processed/all_rules.validated.jsonl"
DEFAULT_OUT = "rag_corpus/autorag/virage_rules/corpus.parquet"
DEFAULT_REPORT_JSON = "rag_corpus/autorag/virage_rules/corpus_export_report.json"
DEFAULT_REPORT_MD = "rag_corpus/autorag/virage_rules/corpus_export_report.md"


def _contents(record: RagRuleRecord) -> str:
    parts = [
        f"Title: {record.title}",
        f"Type: {record.record_type}",
        f"Task: {record.task or ''}",
        f"Chart family: {record.chart_family or ''}",
        f"Applies when: {'; '.join(record.applies_when)}",
        f"Guidance: {'; '.join(record.guidance)}",
        f"Avoid: {'; '.join(record.avoid)}",
        f"Retrieval text: {record.retrieval_text}",
    ]
    return "\n".join(part for part in parts if part.strip())


def export_autorag_corpus(input_path: Path, out_path: Path) -> dict[str, Any]:
    records = [RagRuleRecord.model_validate(raw) for raw in read_jsonl(input_path)]
    rows = []
    for record in records:
        rows.append({
            "doc_id": record.doc_id,
            "contents": _contents(record),
            # AutoRAG expects corpus metadata to be a dictionary, not a JSON string.
            # It will add last_modified_datetime itself when the key is absent.
            "metadata": {
                "record_type": record.record_type,
                "task": record.task,
                "chart_family": record.chart_family,
                "severity": record.severity,
                "source_dataset": record.source.dataset,
            },
        })
    ensure_dir(out_path.parent)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return {"records": len(rows), "output": str(out_path)}


def _md(report: dict[str, Any]) -> str:
    return f"# AutoRAG corpus export\n\nRecords: **{report['records']}**\n\nOutput: `{report['output']}`\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export validated ViRAGE rule records to AutoRAG corpus.parquet.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    args = parser.parse_args()
    root = project_root()
    report = export_autorag_corpus(root / args.input, root / args.output)
    write_json(root / args.report_json, report)
    write_text(root / args.report_md, _md(report))
    print(report)


if __name__ == "__main__":
    main()
