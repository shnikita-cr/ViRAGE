from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .chart_types import canonicalize_chart_type
from .models import VisRAGExample


class VisRAGCorpus:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def load(self) -> list[VisRAGExample]:
        if not self.root.exists():
            raise FileNotFoundError(f"RAG corpus directory does not exist: {self.root}")
        examples = [self._to_example(row, path) for path in sorted(self.root.glob("*.jsonl")) for row in self._read_jsonl(path)]
        if not examples:
            raise ValueError(f"RAG corpus has no examples: {self.root}")
        return examples

    @staticmethod
    def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                text = line.strip()
                if text:
                    try:
                        yield json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc

    @staticmethod
    def _to_example(row: dict[str, Any], path: Path) -> VisRAGExample:
        chart_type = canonicalize_chart_type(row.get("chart_type") or row.get("mark"))
        instruction = row.get("instruction") or row.get("query") or row.get("utterance") or row.get("description")
        if not instruction:
            raise ValueError(f"Corpus row in {path} has no instruction/query/description")
        return VisRAGExample(
            example_id=str(row.get("id") or row.get("example_id") or f"{path.stem}:{abs(hash(instruction))}"),
            source=str(row.get("source") or path.stem),
            corpus=str(row.get("corpus") or path.stem),
            instruction=str(instruction),
            chart_type=chart_type,
            description=row.get("description"),
            keywords=[str(item).lower() for item in row.get("keywords", [])],
            field_roles=dict(row.get("field_roles") or row.get("encoding_roles") or {}),
            transform_types=[str(item) for item in row.get("transform_types", [])],
            spec_template=dict(row.get("spec_template") or row.get("spec") or {}),
            metadata=dict(row.get("metadata") or {}),
        )
