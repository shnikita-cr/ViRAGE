from __future__ import annotations

from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())


import re

_SPACE_RE = re.compile(r"\s+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def compact_text(value: object, *, max_chars: int | None = None) -> str:
    text = _SPACE_RE.sub(" ", str(value or "")).strip()
    if max_chars is not None and len(text) > max_chars:
        return text[: max(0, max_chars - 1)].rstrip() + "…"
    return text


def slugify(value: object, *, max_len: int = 80) -> str:
    text = str(value or "").lower().strip()
    text = _SLUG_RE.sub("_", text).strip("_")
    if not text:
        return "item"
    return text[:max_len].strip("_") or "item"
