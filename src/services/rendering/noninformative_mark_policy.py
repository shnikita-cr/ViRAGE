from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class NonInformativeMarkPolicyResult:
    changes: list[str] = field(default_factory=list)


class NonInformativeMarkPolicy:
    """Remove deterministic zero-information marks produced by generated transforms."""

    @classmethod
    def apply(cls, spec: dict[str, Any], *, data: pd.DataFrame | None) -> NonInformativeMarkPolicyResult:
        if data is None or data.empty or not isinstance(spec, dict):
            return NonInformativeMarkPolicyResult()
        changes: list[str] = []
        if cls._should_filter_zero_missingness(spec, data):
            transforms = spec.setdefault("transform", [])
            if isinstance(transforms, list) and not cls._has_filter(transforms, "missing_percentage"):
                transforms.append({"filter": "datum.missing_percentage > 0"})
                changes.append("Filtered zero missing-percentage fields from missingness view.")
        return NonInformativeMarkPolicyResult(changes=changes)

    @classmethod
    def _should_filter_zero_missingness(cls, spec: dict[str, Any], data: pd.DataFrame) -> bool:
        transforms = spec.get("transform")
        if not isinstance(transforms, list):
            return False
        fold_fields = cls._fold_fields(transforms)
        if not fold_fields:
            return False
        if not cls._encodes_field(spec, "missing_percentage"):
            return False
        available = [field for field in fold_fields if field in data.columns]
        if not available:
            return False
        missing_ratios = [float(data[field].isna().mean()) for field in available]
        nonzero_count = sum(1 for value in missing_ratios if value > 0.0)
        return 0 < nonzero_count < len(available)

    @staticmethod
    def _fold_fields(transforms: list[Any]) -> list[str]:
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            fold = transform.get("fold")
            if isinstance(fold, list):
                return [str(field) for field in fold if isinstance(field, str) and field.strip()]
        return []

    @staticmethod
    def _encodes_field(spec: dict[str, Any], field_name: str) -> bool:
        encoding = spec.get("encoding")
        if not isinstance(encoding, dict):
            return False
        for channel_def in encoding.values():
            if isinstance(channel_def, dict) and channel_def.get("field") == field_name:
                return True
            if isinstance(channel_def, list):
                for item in channel_def:
                    if isinstance(item, dict) and item.get("field") == field_name:
                        return True
        return False

    @staticmethod
    def _has_filter(transforms: list[Any], field_name: str) -> bool:
        needle = field_name.lower()
        for transform in transforms:
            if isinstance(transform, dict) and "filter" in transform:
                value = str(transform.get("filter") or "").lower()
                if needle in value:
                    return True
        return False
