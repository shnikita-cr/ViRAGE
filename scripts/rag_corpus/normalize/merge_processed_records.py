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
from scripts.rag_corpus.common.schemas import ALLOWED_RECORD_TYPES

DEFAULT_INPUT_DIR = "rag_corpus/processed/llm_normalized"
DEFAULT_OUTPUT_DIR = "rag_corpus/processed"


def merge_processed_records(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = {record_type: [] for record_type in sorted(ALLOWED_RECORD_TYPES)}
    all_records: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        for record in read_jsonl(path):
            record_type = record.get("record_type")
            if record_type in by_type:
                by_type[record_type].append(record)
                all_records.append(record)
    for record_type, records in by_type.items():
        write_jsonl(output_dir / f"{record_type}s.jsonl", records)
    write_jsonl(output_dir / "all_rules.jsonl", all_records)
    report = {"total": len(all_records), "by_record_type": {key: len(value) for key, value in by_type.items()}}
    write_json(output_dir / "merge_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LLM-normalized rule records into processed corpus files.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    root = project_root()
    report = merge_processed_records(root / args.input_dir, root / args.output_dir)
    print(report)


if __name__ == "__main__":
    main()
