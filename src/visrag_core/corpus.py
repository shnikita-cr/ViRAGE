from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .chart_types import require_supported_chart_type
from .models import VisRAGExample
from .semantic import normalize_semantic_type

# Field role channels are not restricted to a local subset. Vega-Lite supports many channels
# such as xOffset/yOffset, x2/y2, error channels, geo channels, theta/radius, tooltip, etc.
_CONTROL_FILES = {"manifest.json", "validation_report.json", "lexical_index.json", "corpus_manifest.json"}


class VisRAGCorpus:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def load(self) -> list[VisRAGExample]:
        if not self.root.exists():
            raise FileNotFoundError(f"RAG corpus path does not exist: {self.root}")
        rows: list[tuple[dict[str, Any], Path]] = []
        if self.root.is_file():
            paths = [self.root]
        else:
            paths = sorted(
                path for path in self.root.glob("*")
                if path.is_file()
                and path.name not in _CONTROL_FILES
                and path.suffix.lower() in {".jsonl", ".ndjson", ".json"}
            )
        for path in paths:
            rows.extend((row, path) for row in _read_json_records(path))
        return [self._to_example(row, path) for row, path in rows]

    @staticmethod
    def _to_example(row: dict[str, Any], path: Path) -> VisRAGExample:
        spec_template = row.get("spec_template") or row.get("spec") or row.get("generated_spec") or row.get(
            "revised_spec") or {}
        chart_type = require_supported_chart_type(
            row.get("chart_type") or row.get("mark_type") or row.get("mark") or _mark_from_spec(spec_template)
        )
        instruction = (
                row.get("instruction")
                or row.get("query")
                or row.get("utterance")
                or row.get("user_query")
                or row.get("feedback_for_next_generation")
                or row.get("user_comment")
                or row.get("description")
        )
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Corpus row in {path} has no instruction/query/utterance/description.")
        field_roles = _validate_field_roles(row.get("field_roles") or row.get("encoding_roles") or {}, path)
        if not isinstance(spec_template, dict):
            raise ValueError(f"Corpus row in {path} spec_template/spec must be an object.")
        generated_id = hashlib.sha1(f"{path.name}:{instruction}".encode("utf-8")).hexdigest()[:12]
        return VisRAGExample(
            example_id=str(row.get("id") or row.get("example_id") or f"{path.stem}:{generated_id}"),
            source=str(row.get("source") or path.stem),
            corpus=str(row.get("corpus") or row.get("corpus_type") or path.stem),
            instruction=instruction.strip(),
            chart_type=chart_type,
            description=_description_from_row(row),
            keywords=[str(item).lower() for item in row.get("keywords", row.get("tags", [])) if item],
            field_roles=field_roles,
            transform_types=[str(item) for item in row.get("transform_types", row.get("transforms", [])) if item],
            spec_template=spec_template,
            metadata=_metadata_from_row(row),
        )


def _read_json_records(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            payload, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON corpus file {path}: {exc}") from exc
        yield from _unpack_payload(payload, path)
        index = end


def _unpack_payload(payload: Any, path: Path) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"JSON list in {path} must contain objects.")
            yield item
        return
    if isinstance(payload, dict):
        for key in ("examples", "records", "items", "data", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        raise ValueError(f"JSON key {key!r} in {path} must contain objects.")
                    yield item
                return
        yield payload
        return
    raise ValueError(f"Unsupported JSON corpus root in {path}")


def _description_from_row(row: dict[str, Any]) -> str | None:
    description = row.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    comment = row.get("feedback_for_next_generation") or row.get("user_comment")
    if isinstance(comment, str) and comment.strip():
        return comment.strip()
    judge = row.get("judge_result")
    if isinstance(judge, dict):
        comments = judge.get("improvement_comments") or judge.get("missing_requirements") or []
        if isinstance(comments, list):
            cleaned = [str(item).strip() for item in comments if str(item).strip()]
            if cleaned:
                return " ".join(cleaned)
    return None


def _metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    for key in (
            "record_type",
            "source",
            "status",
            "created_at",
            "run_id",
            "attempt_number",
            "user_comment",
            "requested_regeneration",
            "feedback_weight",
            "rag_usage",
    ):
        if key in row:
            metadata[key] = row[key]
    return metadata


def _validate_field_roles(value: Any, path: Path) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"Corpus row in {path} field_roles must be an object.")
    roles: dict[str, str] = {}
    for channel, role in value.items():
        channel_name = str(channel)
        normalized = normalize_semantic_type(str(role))
        if normalized not in {"quantitative", "temporal", "nominal", "ordinal", "boolean", "geojson"}:
            raise ValueError(f"Corpus row in {path} has unsupported field role value: {role!r}.")
        roles[channel_name] = normalized
    return roles


def _mark_from_spec(spec: Any) -> str | None:
    if not isinstance(spec, dict):
        return None
    mark = spec.get("mark")
    if isinstance(mark, dict):
        value = mark.get("type")
        return str(value) if value else None
    return str(mark) if mark else None
