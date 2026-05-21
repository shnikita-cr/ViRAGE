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

from scripts.rag_corpus.common.io import project_root, read_json, read_jsonl, write_json, write_text

DEFAULT_INPUT = "rag_corpus/runtime/virage_rules.jsonl"
DEFAULT_OUT_JSON = "rag_corpus/runtime/runtime_export_report.json"
DEFAULT_OUT_MD = "rag_corpus/runtime/runtime_export_report.md"


def build_report(input_path: Path) -> dict[str, Any]:
    docs = read_jsonl(input_path)
    by_type: dict[str, int] = {}
    prompt_lengths: list[int] = []
    retrieval_lengths: list[int] = []
    for doc in docs:
        by_type[doc.get("record_type", "unknown")] = by_type.get(doc.get("record_type", "unknown"), 0) + 1
        prompt_lengths.append(len(str(doc.get("prompt_text") or "")))
        retrieval_lengths.append(len(str(doc.get("retrieval_text") or "")))
    return {
        "documents": len(docs),
        "by_record_type": dict(sorted(by_type.items())),
        "avg_prompt_text_chars": round(sum(prompt_lengths) / len(prompt_lengths), 2) if prompt_lengths else 0,
        "avg_retrieval_text_chars": round(sum(retrieval_lengths) / len(retrieval_lengths), 2) if retrieval_lengths else 0,
        "input": str(input_path),
    }


def _md(report: dict[str, Any]) -> str:
    lines = ["# ViRAGE runtime rule corpus report", "", f"Documents: **{report['documents']}**", ""]
    lines.append("| Record type | Count |")
    lines.append("|---|---:|")
    for key, value in report.get("by_record_type", {}).items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append(f"Average prompt text length: **{report['avg_prompt_text_chars']} chars**")
    lines.append(f"Average retrieval text length: **{report['avg_retrieval_text_chars']} chars**")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime corpus report for ViRAGE rule corpus.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    args = parser.parse_args()
    root = project_root()
    report = build_report(root / args.input)
    write_json(root / args.out_json, report)
    write_text(root / args.out_md, _md(report))
    print(report)


if __name__ == "__main__":
    main()
