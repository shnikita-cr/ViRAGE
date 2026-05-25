from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import csv
import io
import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.common.text import compact_text
from scripts.rag_corpus.sources.extract_external_rules_common import (
    DATA_SUFFIXES,
    TEXT_SUFFIXES,
    chunk_text,
    clean_markdown,
    filter_candidate_paths,
    flatten_json,
    iter_candidate_files,
    is_relevant_visualization_source,
    make_source_record,
    read_text_with_fallback,
)
from scripts.rag_corpus.sources.source_registry import QUALITY_CORPUS_BY_ID

VISUAL_QUALITY_TEXT_SUFFIXES = TEXT_SUFFIXES | {".mdx"}
VISUAL_QUALITY_DATA_SUFFIXES = DATA_SUFFIXES | {".csv", ".tsv"}

_TEXT_RECORD_KEYS = (
    "title",
    "name",
    "heading",
    "description",
    "caption",
    "summary",
    "text",
    "alt_text",
    "altText",
    "long_description",
    "chart_type",
    "chartType",
    "task",
    "category",
    "section",
)


def _load_tabular_records(path: Path) -> list[dict[str, Any]]:
    text = read_text_with_fallback(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [dict(row) for row in reader]


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = read_text_with_fallback(path)
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = {"text": line.strip()}
            if isinstance(payload, dict):
                records.append(payload)
            else:
                records.append({"text": compact_text(payload)})
        return records

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            return [{"text": text, "parser_warning": "PyYAML is not installed."}]
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)

    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"text": compact_text(item)} for item in payload]
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "examples", "charts", "captions", "annotations"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"text": compact_text(item)} for item in value]
        return [payload]
    return [{"text": compact_text(payload)}]


def _record_title(payload: dict[str, Any], fallback: str) -> str:
    for key in ("title", "name", "heading", "chart_type", "chartType", "category", "section"):
        value = payload.get(key)
        if value:
            return compact_text(value, max_chars=160)
    return fallback


def _record_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in _TEXT_RECORD_KEYS:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            parts.append(f"{key}: {flatten_json(value, max_chars=1200)}")
    if not parts:
        parts.append(flatten_json(payload, max_chars=4500))
    return compact_text(". ".join(parts), max_chars=4500)


def _preferred_record_type(source_id: str) -> str:
    if source_id in {"ft_visual_vocabulary", "from_data_to_viz", "data_visualisation_catalogue"}:
        return "chart_pattern"
    if source_id in {"w3c_wai_complex_images", "vistext"}:
        return "vlm_readability_rule"
    return "readability_rule"


def extract_visual_quality_text_source(
    input_dir: Path,
    *,
    source_id: str,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    max_records_per_file: int = 12,
) -> list[SourceRecord]:
    if source_id not in QUALITY_CORPUS_BY_ID:
        raise ValueError(f"Unknown quality corpus source: {source_id}")
    source = QUALITY_CORPUS_BY_ID[source_id]
    preferred_record_type = _preferred_record_type(source_id)
    records: list[SourceRecord] = []

    text_files = iter_candidate_files(input_dir, suffixes=VISUAL_QUALITY_TEXT_SUFFIXES)
    text_files = filter_candidate_paths(text_files, base_dir=input_dir, include_paths=include_paths, exclude_paths=exclude_paths)
    for path in text_files:
        try:
            raw_text = read_text_with_fallback(path)
        except Exception:
            continue
        cleaned = clean_markdown(raw_text)
        chunks = chunk_text(cleaned, max_chars=4500, min_chars=180)
        if not chunks and len(cleaned) >= 120:
            chunks = [cleaned[:4500]]
        kept_in_file = 0
        for index, chunk in enumerate(chunks, start=1):
            title = f"{path.stem.replace('_', ' ')} #{index}" if len(chunks) > 1 else path.stem.replace("_", " ")
            keep, reason = is_relevant_visualization_source(
                title=title,
                text=chunk,
                path=path,
                source_dataset=source_id,
                source_type="visual_quality_text",
                min_chars=120,
            )
            if not keep:
                continue
            records.append(make_source_record(
                input_dir=input_dir,
                path=path,
                source_dataset=source_id,
                source_type="visual_quality_text",
                title=title,
                text=chunk,
                preferred_record_type=preferred_record_type,
                record_prefix=source_id,
                metadata={
                    "source_title": source.title,
                    "source_url": source.url,
                    "source_format": source.format,
                    "source_purpose": source.purpose,
                    "file_suffix": path.suffix.lower(),
                    "source_prefilter": reason,
                },
            ))
            kept_in_file += 1
            if kept_in_file >= max_records_per_file:
                break

    data_files = iter_candidate_files(input_dir, suffixes=VISUAL_QUALITY_DATA_SUFFIXES)
    data_files = filter_candidate_paths(data_files, base_dir=input_dir, include_paths=include_paths, exclude_paths=exclude_paths)
    for path in data_files:
        try:
            loaded = _load_tabular_records(path) if path.suffix.lower() in {".csv", ".tsv"} else _load_json_records(path)
        except Exception:
            continue
        kept_in_file = 0
        for idx, payload in enumerate(loaded):
            text = _record_text(payload)
            if len(text) < 120:
                continue
            title = _record_title(payload, path.stem)
            keep, reason = is_relevant_visualization_source(
                title=title,
                text=text,
                path=path,
                source_dataset=source_id,
                source_type="visual_quality_structured_record",
                min_chars=120,
            )
            if not keep:
                continue
            records.append(make_source_record(
                input_dir=input_dir,
                path=path,
                source_dataset=source_id,
                source_type="visual_quality_structured_record",
                title=f"{title} #{idx + 1}" if len(loaded) > 1 else title,
                text=text,
                preferred_record_type=preferred_record_type,
                record_prefix=f"{source_id}_data",
                raw=payload,
                metadata={
                    "source_title": source.title,
                    "source_url": source.url,
                    "source_format": source.format,
                    "source_purpose": source.purpose,
                    "file_suffix": path.suffix.lower(),
                    "record_index": idx,
                    "source_prefilter": reason,
                },
            ))
            kept_in_file += 1
            if kept_in_file >= max_records_per_file:
                break

    return records
