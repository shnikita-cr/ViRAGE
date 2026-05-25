from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import hashlib
import json
from typing import Any


def _normalize_for_json(value: Any) -> Any:
    """Convert arbitrary nested values to a JSON-sortable structure.

    Some source files, especially malformed CSV rows, may contain non-string keys
    such as None. ``json.dumps(..., sort_keys=True)`` cannot sort mixed key
    types. The hasher is used for IDs only, so lossy but deterministic
    normalization is acceptable and safer than letting source extraction fail.
    """
    if isinstance(value, dict):
        normalized_items: list[tuple[str, Any]] = []
        for key, item in value.items():
            key_text = "__none__" if key is None else str(key)
            normalized_items.append((key_text, _normalize_for_json(item)))
        return {key: item for key, item in sorted(normalized_items, key=lambda pair: pair[0])}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, set):
        return sorted((_normalize_for_json(item) for item in value), key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def stable_hash(value: Any, *, length: int = 12) -> str:
    normalized = _normalize_for_json(value)
    text = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
