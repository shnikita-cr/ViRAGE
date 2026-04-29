from __future__ import annotations

from .models import VisRAGRequest
from .semantic import semantic_type_from_role_or_dtype


def map_fields(field_roles: dict[str, str], request: VisRAGRequest) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    missing: list[str] = []
    used: set[str] = set()
    selected = [name for name in request.selected_fields if name]
    columns = request.data_profile.columns
    for channel, role in field_roles.items():
        role_key = semantic_type_from_role_or_dtype(role, role)
        candidates = [
            column.name
            for column in columns
            if semantic_type_from_role_or_dtype(column.role, column.semantic_type, column.raw_dtype) == role_key
        ]
        ordered = [name for name in selected if name in candidates] + [name for name in candidates if
                                                                       name not in selected]
        chosen = next((name for name in ordered if name not in used), None)
        if chosen:
            mapping[channel] = chosen
            used.add(chosen)
        else:
            missing.append(channel)
    return mapping, missing
