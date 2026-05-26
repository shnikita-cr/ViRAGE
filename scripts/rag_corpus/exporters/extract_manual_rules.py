from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.rag_corpus.common.hashing import stable_hash
from scripts.rag_corpus.common.io import project_root, write_jsonl
from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.common.text import compact_text

DEFAULT_INPUT_DIR = "rag_corpus/raw/manual_rules"
DEFAULT_OUTPUT = "rag_corpus/extracted/manual_rules.jsonl"


def _load_file(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or []
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records
    return None


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "rules", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def extract_manual_rules(input_dir: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".json", ".jsonl"}:
            continue
        payload = _load_file(path)
        for idx, item in enumerate(_items(payload)):
            title = compact_text(item.get("title") or item.get("name") or f"Manual rule {idx + 1}")
            text_parts = [
                title,
                compact_text(item.get("description")),
                compact_text(item.get("applies_when")),
                compact_text(item.get("guidance")),
                compact_text(item.get("avoid")),
                compact_text(item.get("retrieval_text")),
            ]
            text = compact_text(". ".join(part for part in text_parts if part), max_chars=4000)
            record_id = item.get("record_id") or f"manual__{stable_hash([str(path), idx, item])}"
            records.append(SourceRecord(
                record_id=record_id,
                source_dataset="manual_rules",
                source_path=str(path),
                source_type="manual_rule_seed",
                title=title,
                text=text,
                task=item.get("task"),
                chart_family=item.get("chart_family"),
                metadata={"preferred_record_type": item.get("record_type")},
                raw=item,
            ))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract manual rule seeds into source records for LLM normalization.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = project_root()
    records = extract_manual_rules(root / args.input_dir)
    write_jsonl(root / args.output, [record.model_dump() for record in records])
    print(f"Wrote {len(records)} source records to {args.output}")


if __name__ == "__main__":
    main()
