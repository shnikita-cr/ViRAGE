from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import hashlib
import json
from collections.abc import Mapping, Sequence, Set
from typing import Any


def _normalize_for_hash(value: Any) -> Any:
    """Return a JSON-stable representation for arbitrary nested Python values.

    CSV loaders can produce dictionaries with ``None`` keys when a row has more
    columns than the header. ``json.dumps(..., sort_keys=True)`` cannot sort a
    mixture of ``None`` and ``str`` keys, so we normalize mappings into a sorted
    list of key/value pairs with stringified keys.
    """
    if isinstance(value, Mapping):
        normalized_items = [
            (str(key), _normalize_for_hash(item_value))
            for key, item_value in value.items()
        ]
        normalized_items.sort(key=lambda item: item[0])
        return {"__mapping__": normalized_items}
    if isinstance(value, (str, bytes)):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, Set):
        return sorted(_normalize_for_hash(item) for item in value)
    return value


def stable_hash(value: Any, *, length: int = 12) -> str:
    normalized = _normalize_for_hash(value)
    text = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]
