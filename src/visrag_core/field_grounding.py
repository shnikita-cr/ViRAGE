from __future__ import annotations

from .models import VisRAGColumnProfile, VisRAGRequest
from .semantic import semantic_type_from_role_or_dtype


STRICT_SELECTED_FIELDS_POLICY = "strict"


def map_fields(field_roles: dict[str, str], request: VisRAGRequest) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    missing: list[str] = []
    used: set[str] = set()
    selected = [name for name in request.selected_fields if name]
    columns = _candidate_columns(request)

    for channel, role in field_roles.items():
        role_key = semantic_type_from_role_or_dtype(role, role)
        candidates = [
            column.name
            for column in columns
            if semantic_type_from_role_or_dtype(column.role, column.semantic_type, column.raw_dtype) == role_key
        ]
        ordered = [name for name in selected if name in candidates] + [name for name in candidates if name not in selected]
        chosen = next((name for name in ordered if name not in used), None)

        if chosen:
            mapping[channel] = chosen
            used.add(chosen)
        else:
            missing.append(channel)

    return mapping, missing


def _candidate_columns(request: VisRAGRequest) -> list[VisRAGColumnProfile]:
    policy = str(getattr(request, "selected_fields_policy", "prefer") or "prefer").strip().lower()
    selected = {name for name in request.selected_fields if name}

    if policy != STRICT_SELECTED_FIELDS_POLICY or not selected:
        return list(request.data_profile.columns)

    return [column for column in request.data_profile.columns if column.name in selected]
