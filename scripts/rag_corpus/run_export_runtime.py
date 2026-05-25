from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import argparse
import json

from scripts.rag_corpus.common.io import project_root, write_json
from scripts.rag_corpus.common.progress import StageProgress

CORPUS_PROFILES = {
    "validated": "rag_corpus/processed/all_rules.validated.jsonl",
    "filtered": "rag_corpus/processed/all_rules.filtered.jsonl",
    "embedding_deduped": "rag_corpus/processed/all_rules.embedding_deduped.jsonl",
    "deduped": "rag_corpus/processed/all_rules.deduped.jsonl",
}


def resolve_input_path(profile: str, input_path: str | None) -> str:
    if input_path:
        return input_path
    return CORPUS_PROFILES[profile]
from scripts.rag_corpus.runtime.build_runtime_report import build_report
from scripts.rag_corpus.runtime.export_runtime_rules import export_runtime_rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ViRAGE processed rule corpus to runtime JSONL.")
    parser.add_argument("--input", default=None, help="Explicit processed JSONL path. Overrides --profile.")
    parser.add_argument("--profile", choices=sorted(CORPUS_PROFILES), default="validated", help="Processed corpus profile to export when --input is not provided.")
    parser.add_argument("--output", default="rag_corpus/runtime/virage_rules.jsonl")
    args = parser.parse_args()
    root = project_root()
    progress = StageProgress("runtime-export", total=2)
    export_report = export_runtime_rules(root / resolve_input_path(args.profile, args.input), root / args.output)
    progress.update(extra=f"documents={export_report.get('documents', 0)}")
    runtime_report = build_report(root / args.output)
    progress.update(extra="report written")
    progress.finish()
    report = {"export": export_report, "runtime": runtime_report}
    write_json(root / "rag_corpus/runtime/runtime_export_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
