from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.rag_corpus.common.text.normalization import compact_text
from scripts.rag_corpus.exporters.common.text import DEFAULT_MAX_TEXT_CHARS, read_text_strict


def chunk_text(text: str, *, max_chars: int = DEFAULT_MAX_TEXT_CHARS, min_chars: int = 180) -> list[str]:
    text = compact_text(text, max_chars=None)
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if split_at > start + min_chars:
                end = split_at + 1
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        start = end
    return chunks


def split_markdown_sections(text: str, *, min_chars: int = 180, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    title = ""
    parts: list[str] = []

    def flush() -> None:
        nonlocal parts
        body = compact_text(" ".join(parts), max_chars=None)
        if len(body) >= min_chars:
            for chunk in chunk_text(body, max_chars=max_chars, min_chars=min_chars):
                sections.append((title or "Markdown section", chunk))
        parts = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            title = stripped.lstrip("#").strip()
        elif stripped:
            parts.append(stripped)
    flush()
    return sections


def flatten_json(value: Any, *, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    if isinstance(value, dict):
        parts = [f"{key}: {flatten_json(item, max_chars=500)}" for key, item in value.items() if item not in (None, "", [], {})]
        return compact_text("; ".join(parts), max_chars=max_chars)
    if isinstance(value, list):
        return compact_text("; ".join(flatten_json(item, max_chars=500) for item in value[:20]), max_chars=max_chars)
    return compact_text(value, max_chars=max_chars)


def load_json_like_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    text = read_text_strict(path)
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL record in {path} must be an object")
                records.append(payload)
        return records
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            if not all(isinstance(item, dict) for item in payload):
                raise ValueError(f"JSON list in {path} must contain only objects")
            return payload
        if isinstance(payload, dict):
            for key in ("records", "data", "items", "examples", "tasks", "rules", "constraints"):
                value = payload.get(key)
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    return value
            return [payload]
    raise ValueError(f"Unsupported JSON source shape in {path}")
