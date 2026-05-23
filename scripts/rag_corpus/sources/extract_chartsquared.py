from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.rag_corpus.common.hashing import stable_hash
from scripts.rag_corpus.common.io import project_root, write_jsonl
from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.common.text import compact_text

DEFAULT_INPUT_DIR = "rag_corpus/raw/chartsquared"
DEFAULT_OUTPUT = "rag_corpus/extracted/chartsquared.jsonl"
TEXT_KEYS = (
    "question", "query", "instruction", "request", "prompt", "feedback", "critique", "comment", "answer",
    "task", "purpose", "audience", "chart_type", "chart", "caption", "description"
)


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key in TEXT_KEYS:
            if key in value and value[key]:
                parts.append(f"{key}: {_flatten_text(value[key])}")
        if parts:
            return ". ".join(parts)
        return compact_text(value, max_chars=2500)
    if isinstance(value, list):
        return "; ".join(_flatten_text(item) for item in value[:20])
    return compact_text(value)


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        records.append(value)
                except json.JSONDecodeError:
                    records.append({"text": line.strip()})
    return records


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_records(path)
    if suffix == ".jsonl":
        return _load_jsonl_records(path)
    if suffix == ".csv":
        return _load_csv_records(path)
    return []


def extract_chartsquared(input_dir: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        for idx, item in enumerate(_load_records(path)):
            text = compact_text(_flatten_text(item), max_chars=4000)
            if not text:
                continue
            record_id = str(item.get("id") or item.get("record_id") or f"chartsquared__{stable_hash([str(path), idx, item])}")
            records.append(SourceRecord(
                record_id=record_id,
                source_dataset="chartsquared",
                source_path=str(path),
                source_type="chart_feedback_or_task",
                title=compact_text(item.get("title") or item.get("question") or item.get("query") or "ChartSquared record"),
                text=text,
                task=item.get("task"),
                chart_family=item.get("chart_type") or item.get("chart"),
                metadata={"relative_source_path": str(path.relative_to(input_dir)) if path.is_relative_to(input_dir) else str(path)},
                raw=item,
            ))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ChartSquared/C-2 records into source records for LLM normalization.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = project_root()
    records = extract_chartsquared(root / args.input_dir)
    write_jsonl(root / args.output, [record.model_dump() for record in records])
    print(f"Wrote {len(records)} source records to {args.output}")


if __name__ == "__main__":
    main()
