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

from scripts.rag_corpus.common.io import project_root, read_jsonl, write_json, write_jsonl, write_text
from scripts.rag_corpus.common.schemas import RagRuleRecord
from scripts.rag_corpus.common.validation import summarize_rule_records, validate_rule_records

DEFAULT_INPUT = "rag_corpus/processed/all_rules.deduped.jsonl"
DEFAULT_VALID = "rag_corpus/processed/all_rules.validated.jsonl"
DEFAULT_REJECTED = "rag_corpus/processed/rejected_records.jsonl"
DEFAULT_REPORT_JSON = "rag_corpus/processed/processing_report.json"
DEFAULT_REPORT_MD = "rag_corpus/processed/processing_report.md"


def _markdown(summary: dict[str, Any], rejected_count: int) -> str:
    lines = ["# RAG rule corpus processing report", ""]
    lines.append(f"Total valid records: **{summary.get('total', 0)}**")
    lines.append(f"Rejected records: **{rejected_count}**")
    lines.append("")
    lines.append("## By record type")
    lines.append("")
    lines.append("| Record type | Count |")
    lines.append("|---|---:|")
    for key, value in summary.get("by_record_type", {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append("## By source")
    lines.append("")
    lines.append("| Source dataset | Count |")
    lines.append("|---|---:|")
    for key, value in summary.get("by_source_dataset", {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def validate_processed(input_path: Path, valid_path: Path, rejected_path: Path, report_json: Path, report_md: Path) -> dict[str, Any]:
    raw = read_jsonl(input_path)
    valid, rejected = validate_rule_records(raw)
    write_jsonl(valid_path, [record.model_dump() for record in valid])
    if rejected:
        write_jsonl(rejected_path, rejected, append=True)
    summary = summarize_rule_records(valid)
    summary["rejected"] = len(rejected)
    write_json(report_json, summary)
    write_text(report_md, _markdown(summary, len(rejected)))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed ViRAGE rule records.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--valid", default=DEFAULT_VALID)
    parser.add_argument("--rejected", default=DEFAULT_REJECTED)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    args = parser.parse_args()
    root = project_root()
    summary = validate_processed(root / args.input, root / args.valid, root / args.rejected, root / args.report_json, root / args.report_md)
    print(summary)


if __name__ == "__main__":
    main()
