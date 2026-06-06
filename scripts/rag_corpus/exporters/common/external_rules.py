from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
from pathlib import Path
from typing import Any
from scripts.rag_corpus.common.text.hashing import stable_hash
from scripts.rag_corpus.common.io import project_root, write_jsonl
from scripts.rag_corpus.common.models.schemas import SourceRecord
from scripts.rag_corpus.common.text.normalization import compact_text
from scripts.rag_corpus.exporters.common.filters import is_relevant_visualization_source
from scripts.rag_corpus.exporters.common.html import extract_html_sections
from scripts.rag_corpus.exporters.common.structured import chunk_text, flatten_json, load_json_like_records, split_markdown_sections
from scripts.rag_corpus.exporters.common.text import CODE_SUFFIXES, DATA_SUFFIXES, DEFAULT_MAX_RECORDS_PER_FILE, DEFAULT_MAX_TEXT_CHARS, TEXT_SUFFIXES, clean_extracted_text, clean_markdown, filter_candidate_paths, iter_candidate_files, keyword_score, path_pattern_score, read_text_strict

def make_source_record(*, input_dir: Path, path: Path, source_dataset: str, source_type: str, title: str, text: str, preferred_record_type: str, record_prefix: str, raw: dict[str, Any] | None=None, metadata: dict[str, Any] | None=None) -> SourceRecord:
    relative = str(path.relative_to(input_dir)) if path.is_relative_to(input_dir) else str(path)
    normalized_text = compact_text(text, max_chars=DEFAULT_MAX_TEXT_CHARS)
    return SourceRecord(record_id=f'{record_prefix}__{stable_hash([relative, title, normalized_text])}', source_dataset=source_dataset, source_path=str(path), source_type=source_type, title=compact_text(title or path.stem.replace('_', ' '), max_chars=180), text=normalized_text, metadata={'relative_source_path': relative, 'preferred_record_type': preferred_record_type, 'extractor': record_prefix, **(metadata or {})}, raw=raw or {'title': title, 'text': normalized_text})

def extract_markdown_like(input_dir: Path, *, source_dataset: str, record_prefix: str, preferred_record_type: str, source_type: str, suffixes: set[str] | None=None, include_keywords: list[str] | None=None, exclude_keywords: list[str] | None=None, include_paths: list[str] | None=None, exclude_paths: list[str] | None=None, max_records_per_file: int=DEFAULT_MAX_RECORDS_PER_FILE) -> list[SourceRecord]:
    files = iter_candidate_files(input_dir, suffixes=suffixes or TEXT_SUFFIXES)
    files = filter_candidate_paths(files, base_dir=input_dir, include_paths=include_paths, exclude_paths=exclude_paths)
    if include_keywords:
        files = [path for path in files if keyword_score(path, include_keywords) > 0 or path.name.lower() in {'readme.md', 'index.md'}]
    if exclude_keywords:
        files = [path for path in files if keyword_score(path, exclude_keywords) == 0]
    records: list[SourceRecord] = []
    for path in files:
        try:
            text = read_text_strict(path)
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as exc:
            raise RuntimeError(f'Cannot read source file {path}: {exc}') from exc
        sections = _sections_for_text_source(path, text)
        records.extend(_records_from_sections(input_dir=input_dir, path=path, sections=sections, source_dataset=source_dataset, source_type=source_type, preferred_record_type=preferred_record_type, record_prefix=record_prefix, max_records=max_records_per_file))
    return records

def _sections_for_text_source(path: Path, text: str) -> list[tuple[str, str]]:
    if path.suffix.lower() in {'.html', '.htm'}:
        return extract_html_sections(text, default_title=path.stem.replace('_', ' '))
    sections = split_markdown_sections(text)
    if sections:
        return sections
    cleaned = clean_markdown(text)
    return [(path.stem.replace('_', ' '), chunk) for chunk in chunk_text(cleaned)]

def _records_from_sections(*, input_dir: Path, path: Path, sections: list[tuple[str, str]], source_dataset: str, source_type: str, preferred_record_type: str, record_prefix: str, max_records: int) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for title, body in sections:
        keep, reason = is_relevant_visualization_source(title=title, text=body, path=path, source_dataset=source_dataset, source_type=source_type)
        if not keep:
            continue
        records.append(make_source_record(input_dir=input_dir, path=path, source_dataset=source_dataset, source_type=source_type, title=title, text=body, preferred_record_type=preferred_record_type, record_prefix=record_prefix, metadata={'file_suffix': path.suffix.lower(), 'source_prefilter': reason}))
        if len(records) >= max_records:
            break
    return records

def extract_json_like(input_dir: Path, *, source_dataset: str, record_prefix: str, preferred_record_type: str, source_type: str, include_keywords: list[str] | None=None, include_paths: list[str] | None=None, exclude_paths: list[str] | None=None, max_records_per_file: int=DEFAULT_MAX_RECORDS_PER_FILE) -> list[SourceRecord]:
    files = iter_candidate_files(input_dir, suffixes={'.json', '.jsonl'})
    files = filter_candidate_paths(files, base_dir=input_dir, include_paths=include_paths, exclude_paths=exclude_paths)
    if include_keywords:
        files = [path for path in files if keyword_score(path, include_keywords) > 0]
    records: list[SourceRecord] = []
    for path in files:
        try:
            loaded = load_json_like_records(path)
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as exc:
            raise RuntimeError(f'Cannot parse structured source file {path}: {exc}') from exc
        records.extend(_records_from_json_items(input_dir=input_dir, path=path, items=loaded, source_dataset=source_dataset, source_type=source_type, preferred_record_type=preferred_record_type, record_prefix=record_prefix, max_records=max_records_per_file))
    return records

def _records_from_json_items(*, input_dir: Path, path: Path, items: list[dict[str, Any]], source_dataset: str, source_type: str, preferred_record_type: str, record_prefix: str, max_records: int) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for idx, payload in enumerate(items):
        text = flatten_json(payload)
        if len(text) < 120:
            continue
        title = compact_text(payload.get('title') or payload.get('name') or payload.get('task') or path.stem, max_chars=160)
        keep, reason = is_relevant_visualization_source(title=title, text=text, path=path, source_dataset=source_dataset, source_type=source_type, min_chars=120)
        if not keep:
            continue
        records.append(make_source_record(input_dir=input_dir, path=path, source_dataset=source_dataset, source_type=source_type, title=f'{title} #{idx + 1}' if len(items) > 1 else title, text=text, preferred_record_type=preferred_record_type, record_prefix=record_prefix, raw=payload, metadata={'file_suffix': path.suffix.lower(), 'record_index': idx, 'source_prefilter': reason}))
        if len(records) >= max_records:
            break
    return records

def write_extractor_cli(*, description: str, default_input_dir: str, default_output: str, extractor) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input-dir', default=default_input_dir)
    parser.add_argument('--output', default=default_output)
    parser.add_argument('--include-path', action='append', default=[])
    parser.add_argument('--exclude-path', action='append', default=[])
    args = parser.parse_args()
    root = project_root()
    try:
        records = extractor(root / args.input_dir, include_paths=args.include_path, exclude_paths=args.exclude_path)
    except TypeError:
        records = extractor(root / args.input_dir)
    write_jsonl(root / args.output, [record.model_dump() for record in records])
    logger.info(f'Wrote {len(records)} source records to {args.output}')
__all__ = ['CODE_SUFFIXES', 'DATA_SUFFIXES', 'TEXT_SUFFIXES', 'chunk_text', 'clean_extracted_text', 'clean_markdown', 'extract_html_sections', 'extract_json_like', 'extract_markdown_like', 'filter_candidate_paths', 'flatten_json', 'is_relevant_visualization_source', 'iter_candidate_files', 'keyword_score', 'make_source_record', 'path_pattern_score', 'read_text_strict', 'split_markdown_sections', 'write_extractor_cli']
