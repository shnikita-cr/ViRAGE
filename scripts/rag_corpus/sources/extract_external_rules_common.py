from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from scripts.rag_corpus.common.hashing import stable_hash
from scripts.rag_corpus.common.io import project_root, write_jsonl
from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.common.text import compact_text

TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "cp1252", "latin-1")
TEXT_SUFFIXES = {".md", ".txt", ".rst", ".html", ".htm", ".rmd"}
DATA_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml"}
CODE_SUFFIXES = {".lp", ".asp", ".pl", ".py", ".ts", ".js"}
DEFAULT_MAX_RECORDS_PER_FILE = 8
DEFAULT_MAX_TEXT_CHARS = 4500

_SKIP_PARTS = {
    ".git", "node_modules", "dist", "build", "venv", ".venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", "target", "coverage", ".next", ".cache", "vendor",
}

_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz", ".tar", ".tgz",
    ".parquet", ".feather", ".xlsx", ".xls", ".sqlite", ".db", ".pkl", ".pickle",
}


_VISUALIZATION_TERMS = {
    "chart", "plot", "visual", "visualization", "visualisation", "graph", "axis", "axes",
    "legend", "tooltip", "label", "mark", "encoding", "channel", "aggregate", "aggregation",
    "bin", "histogram", "scatter", "line", "bar", "map", "choropleth", "heatmap", "boxplot",
    "violin", "distribution", "correlation", "trend", "ranking", "comparison", "compare",
    "category", "categorical", "quantitative", "temporal", "time", "date", "color", "size",
    "facet", "sort", "filter", "scale", "readability", "overplot", "outlier", "data",
}

_SOURCE_NOISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(load|install|import|require)\s+(the\s+)?(package|library|module|dependency|dependencies)\b",
        r"\b(pip|npm|yarn|conda|poetry|cargo|docker|webpack|vite)\b",
        r"\b(unit\s+test|test\s+suite|pytest|jest|coverage|ci|github\s+action)\b",
        r"\b(provider\s+tos|terms\s+of\s+service|license|copyright)\b",
        r"\b(api\s+key|token|authentication|authorization|login|account)\b",
        r"\b(html|css|javascript|typescript|python|r\s+code|script|function|class|method)\b.*\b(example|implementation|utility|helper)\b",
        r"\b(file|folder|directory|repository|repo|readme)\b.*\b(structure|organization|layout)\b",
        r"\b(organize|name|rename|document|describe)\b.*\b(script|function|test|file|module)\b",
        r"\b(release|changelog|contributing|contribution|issue|pull\s+request)\b",
        r"\b(load\s+required\s+libraries|check\s+provider\s+tos|organize\s+.*scripts|name\s+functions\s+clearly|describe\s+tests\s+clearly)\b",
    ]
]

_TITLE_NOISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^(installation|install|setup|getting started|quick start|usage|api|development|contributing|license|tests?|examples?)$",
        r"^(load required libraries|provider tos|organize .*scripts|name functions clearly|describe tests clearly)$",
    ]
]

def _tokenize_for_relevance(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()))

def is_relevant_visualization_source(
    *,
    title: str,
    text: str,
    path: Path | None = None,
    source_dataset: str = "",
    source_type: str = "",
    min_chars: int = 160,
) -> tuple[bool, str]:
    """Return whether a raw source fragment is worth sending to LLM normalization.

    The filter is intentionally conservative for true visualization guidance and
    strict for repository/documentation noise. It prevents expensive LLM calls on
    setup pages, source-code organization notes and short headings that cannot be
    converted into reliable RAG rules.
    """
    title_text = compact_text(title or "", max_chars=240)
    body = compact_text(text or "", max_chars=6000)
    combined = f"{title_text}. {body}".strip()
    if len(body) < min_chars:
        return False, "too_short_raw_text"
    lower_title = title_text.lower().strip()
    for pattern in _TITLE_NOISE_PATTERNS:
        if pattern.search(lower_title):
            return False, "noise_title"
    for pattern in _SOURCE_NOISE_PATTERNS:
        if pattern.search(combined):
            # Keep if the same fragment strongly discusses visualization design.
            tokens = _tokenize_for_relevance(combined)
            if len(tokens & _VISUALIZATION_TERMS) < 4:
                return False, "documentation_or_code_noise"
    tokens = _tokenize_for_relevance(combined)
    vis_hits = len(tokens & _VISUALIZATION_TERMS)
    if vis_hits < 2:
        return False, "not_visualization_guidance"
    path_text = str(path or "").replace("\\", "/").lower()
    bad_path_parts = (
        "/test", "/tests", "/example", "/examples", "/demo", "/demos", "/doc/api",
        "/site", "/website", "/assets", "/static", "/css", "/js", "/scripts",
    )
    if any(part in path_text for part in bad_path_parts) and vis_hits < 4:
        return False, "low_value_repository_area"
    return True, "kept"


def read_text_with_fallback(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return path.read_text(encoding="utf-8", errors="replace")


def is_ignored_path(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & _SKIP_PARTS:
        return True
    return path.suffix.lower() in _BINARY_SUFFIXES


def iter_candidate_files(input_dir: Path, *, suffixes: set[str], max_file_size: int = 1_500_000) -> list[Path]:
    if not input_dir.exists():
        return []
    result: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or is_ignored_path(path):
            continue
        if path.suffix.lower() not in suffixes:
            continue
        try:
            if path.stat().st_size > max_file_size:
                continue
        except OSError:
            continue
        result.append(path)
    return result


def clean_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_markdown_sections(text: str, *, min_chars: int = 180, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "Document overview"
    current_lines: list[str] = []
    heading_re = re.compile(r"^\s{0,3}#{1,4}\s+(.+?)\s*$")
    for line in lines:
        match = heading_re.match(line)
        if match:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = compact_text(match.group(1), max_chars=160)
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    result: list[tuple[str, str]] = []
    for title, section_lines in sections:
        body = clean_markdown("\n".join(section_lines))
        if len(body) < min_chars:
            continue
        if len(body) <= max_chars:
            result.append((title, body))
            continue
        chunks = chunk_text(body, max_chars=max_chars, min_chars=min_chars)
        for idx, chunk in enumerate(chunks, start=1):
            result.append((f"{title} #{idx}", chunk))
    return result


def chunk_text(text: str, *, max_chars: int = DEFAULT_MAX_TEXT_CHARS, min_chars: int = 180) -> list[str]:
    text = compact_text(text)
    if not text:
        return []
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-ZА-Я])", text) if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= max_chars:
            current = f"{current} {paragraph}".strip()
        else:
            if len(current) >= min_chars:
                chunks.append(current)
            current = paragraph[:max_chars]
    if len(current) >= min_chars:
        chunks.append(current)
    return chunks or ([text[:max_chars]] if len(text) >= min_chars else [])


def flatten_json(value: Any, *, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        priority_keys = [
            "title", "name", "task", "description", "question", "query", "instruction", "prompt",
            "criteria", "feedback", "purpose", "audience", "chart", "chart_type", "type", "text",
        ]
        used: set[str] = set()
        for key in priority_keys:
            if key in value and value[key] not in (None, "", [], {}):
                used.add(key)
                parts.append(f"{key}: {flatten_json(value[key], max_chars=max_chars // 2)}")
        for key, item in value.items():
            if key in used or item in (None, "", [], {}):
                continue
            if len(parts) >= 16:
                break
            parts.append(f"{key}: {flatten_json(item, max_chars=500)}")
        return compact_text(". ".join(parts), max_chars=max_chars)
    if isinstance(value, list):
        return compact_text("; ".join(flatten_json(item, max_chars=500) for item in value[:20]), max_chars=max_chars)
    return compact_text(value, max_chars=max_chars)


def load_json_like_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    text = read_text_with_fallback(path)
    if suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                records.append({"text": line.strip()})
                continue
            if isinstance(payload, dict):
                records.append(payload)
            else:
                records.append({"text": flatten_json(payload)})
        return records
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [item if isinstance(item, dict) else {"text": flatten_json(item)} for item in payload]
        if isinstance(payload, dict):
            for key in ("records", "data", "items", "examples", "tasks", "rules", "constraints"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item if isinstance(item, dict) else {"text": flatten_json(item)} for item in value]
            return [payload]
    return []


def keyword_score(path: Path, keywords: Iterable[str]) -> int:
    haystack = str(path).replace("\\", "/").lower()
    return sum(1 for keyword in keywords if keyword.lower() in haystack)


def make_source_record(
    *,
    input_dir: Path,
    path: Path,
    source_dataset: str,
    source_type: str,
    title: str,
    text: str,
    preferred_record_type: str,
    record_prefix: str,
    raw: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SourceRecord:
    relative = str(path.relative_to(input_dir)) if path.is_relative_to(input_dir) else str(path)
    normalized_text = compact_text(text, max_chars=DEFAULT_MAX_TEXT_CHARS)
    return SourceRecord(
        record_id=f"{record_prefix}__{stable_hash([relative, title, normalized_text])}",
        source_dataset=source_dataset,
        source_path=str(path),
        source_type=source_type,
        title=compact_text(title or path.stem.replace("_", " "), max_chars=180),
        text=normalized_text,
        metadata={
            "relative_source_path": relative,
            "preferred_record_type": preferred_record_type,
            "extractor": record_prefix,
            **(metadata or {}),
        },
        raw=raw or {"title": title, "text": normalized_text},
    )


def extract_markdown_like(
    input_dir: Path,
    *,
    source_dataset: str,
    record_prefix: str,
    preferred_record_type: str,
    source_type: str,
    suffixes: set[str] | None = None,
    include_keywords: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
    max_records_per_file: int = DEFAULT_MAX_RECORDS_PER_FILE,
) -> list[SourceRecord]:
    suffixes = suffixes or TEXT_SUFFIXES
    include_keywords = include_keywords or []
    exclude_keywords = exclude_keywords or []
    files = iter_candidate_files(input_dir, suffixes=suffixes)
    if include_keywords:
        files = [path for path in files if keyword_score(path, include_keywords) > 0 or path.name.lower() in {"readme.md", "index.md"}]
    if exclude_keywords:
        files = [path for path in files if keyword_score(path, exclude_keywords) == 0]
    records: list[SourceRecord] = []
    for path in files:
        try:
            text = read_text_with_fallback(path)
        except Exception:
            continue
        sections = split_markdown_sections(text)
        if not sections:
            cleaned = clean_markdown(text)
            sections = [(path.stem.replace("_", " "), chunk) for chunk in chunk_text(cleaned)]
        kept_in_file = 0
        for title, body in sections:
            keep, reason = is_relevant_visualization_source(
                title=title,
                text=body,
                path=path,
                source_dataset=source_dataset,
                source_type=source_type,
            )
            if not keep:
                continue
            records.append(make_source_record(
                input_dir=input_dir,
                path=path,
                source_dataset=source_dataset,
                source_type=source_type,
                title=title,
                text=body,
                preferred_record_type=preferred_record_type,
                record_prefix=record_prefix,
                metadata={"file_suffix": path.suffix.lower(), "source_prefilter": reason},
            ))
            kept_in_file += 1
            if kept_in_file >= max_records_per_file:
                break
    return records


def extract_json_like(
    input_dir: Path,
    *,
    source_dataset: str,
    record_prefix: str,
    preferred_record_type: str,
    source_type: str,
    include_keywords: list[str] | None = None,
    max_records_per_file: int = DEFAULT_MAX_RECORDS_PER_FILE,
) -> list[SourceRecord]:
    include_keywords = include_keywords or []
    files = iter_candidate_files(input_dir, suffixes={".json", ".jsonl"})
    if include_keywords:
        files = [path for path in files if keyword_score(path, include_keywords) > 0]
    records: list[SourceRecord] = []
    for path in files:
        try:
            loaded = load_json_like_records(path)
        except Exception:
            continue
        kept_in_file = 0
        for idx, payload in enumerate(loaded):
            text = flatten_json(payload)
            if len(text) < 120:
                continue
            title = compact_text(payload.get("title") or payload.get("name") or payload.get("task") or path.stem, max_chars=160)
            keep, reason = is_relevant_visualization_source(
                title=title,
                text=text,
                path=path,
                source_dataset=source_dataset,
                source_type=source_type,
                min_chars=120,
            )
            if not keep:
                continue
            records.append(make_source_record(
                input_dir=input_dir,
                path=path,
                source_dataset=source_dataset,
                source_type=source_type,
                title=f"{title} #{idx + 1}" if len(loaded) > 1 else title,
                text=text,
                preferred_record_type=preferred_record_type,
                record_prefix=record_prefix,
                raw=payload,
                metadata={"file_suffix": path.suffix.lower(), "record_index": idx, "source_prefilter": reason},
            ))
            kept_in_file += 1
            if kept_in_file >= max_records_per_file:
                break
    return records


def write_extractor_cli(
    *,
    description: str,
    default_input_dir: str,
    default_output: str,
    extractor,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input-dir", default=default_input_dir)
    parser.add_argument("--output", default=default_OUTPUT if False else default_output)
    args = parser.parse_args()
    root = project_root()
    records = extractor(root / args.input_dir)
    write_jsonl(root / args.output, [record.model_dump() for record in records])
    print(f"Wrote {len(records)} source records to {args.output}")
