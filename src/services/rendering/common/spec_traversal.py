from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_COMPOSITION_KEYS = ("layer", "hconcat", "vconcat", "concat")


def iter_unit_specs(spec: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(spec, dict):
        return
    if isinstance(spec.get("encoding"), dict) or "mark" in spec:
        yield spec
    for key in _COMPOSITION_KEYS:
        value = spec.get(key)
        if isinstance(value, list):
            for item in value:
                yield from iter_unit_specs(item)
    nested = spec.get("spec")
    if isinstance(nested, dict):
        yield from iter_unit_specs(nested)
