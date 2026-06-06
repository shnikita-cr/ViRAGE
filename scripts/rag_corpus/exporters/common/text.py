from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "cp1252", "latin-1")
TEXT_SUFFIXES = {".md", ".txt", ".rst", ".html", ".htm", ".rmd"}
DATA_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml"}
CODE_SUFFIXES = {".lp", ".asp", ".pl", ".py", ".ts", ".js"}
DEFAULT_MAX_RECORDS_PER_FILE = 8
DEFAULT_MAX_TEXT_CHARS = 4500

_SKIP_PARTS = {".git", "node_modules", "dist", "build", "venv", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", "target", "coverage", ".next", ".cache", "vendor"}
_BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz", ".tar", ".tgz", ".parquet", ".feather", ".xlsx", ".xls", ".sqlite", ".db", ".pkl", ".pickle"}


def read_text_strict(path: Path) -> str:
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
    return bool(lower_parts & _SKIP_PARTS) or path.suffix.lower() in _BINARY_SUFFIXES


def iter_candidate_files(input_dir: Path, *, suffixes: set[str], max_file_size: int = 1_500_000) -> list[Path]:
    if not input_dir.exists():
        return []
    result: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or is_ignored_path(path) or path.suffix.lower() not in suffixes:
            continue
        try:
            if path.stat().st_size <= max_file_size:
                result.append(path)
        except OSError:
            continue
    return result


def clean_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_extracted_text(text: str, *, max_chars: int | None = None) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars is not None and len(text) > max_chars:
        return text[: max(0, max_chars - 1)].rstrip() + "…"
    return text


def keyword_score(path: Path, keywords: Iterable[str]) -> int:
    haystack = str(path).replace("\\", "/").lower()
    return sum(1 for keyword in keywords if keyword.lower() in haystack)


def path_pattern_score(path: Path, patterns: Iterable[str], *, base_dir: Path | None = None) -> int:
    haystack = path.relative_to(base_dir).as_posix().lower() if base_dir and path.is_relative_to(base_dir) else path.as_posix().lower()
    return sum(1 for pattern in patterns if pattern.lower().replace("\\", "/") in haystack)


def filter_candidate_paths(files: list[Path], *, base_dir: Path, include_paths: Iterable[str] | None = None, exclude_paths: Iterable[str] | None = None) -> list[Path]:
    result = files
    if include_paths:
        result = [path for path in result if path_pattern_score(path, include_paths, base_dir=base_dir) > 0]
    if exclude_paths:
        result = [path for path in result if path_pattern_score(path, exclude_paths, base_dir=base_dir) == 0]
    return result
