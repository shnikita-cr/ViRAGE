from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import pandas as pd

from src.services.rendering.common.spec_traversal import iter_unit_specs


@dataclass(frozen=True)
class AxisDomainPolicyResult:
    changes: list[str] = field(default_factory=list)


class AxisDomainPolicy:
    """Normalize quantitative axis domains before rendering.

    The policy is deterministic and data-aware. It keeps true baselines for counts and
    baseline-dependent encodings, but disables forced zero domains for distribution,
    scatter, point and line views where a zero baseline would waste most of the plot area.
    """

    POSITION_CHANNELS = {"x", "y"}
    ZERO_BASELINE_AGGREGATES = {"count", "valid", "missing", "distinct"}
    ZERO_BASELINE_FIELD_TOKENS = {"count", "records", "frequency", "missing_count"}
    PROPORTION_FIELD_TOKENS = {"ratio", "rate", "percentage", "percent", "share", "proportion"}
    BASELINE_MARKS = {"bar", "area"}
    DISTRIBUTION_MARKS = {"boxplot", "point", "circle", "square", "tick", "line", "trail", "rule", "errorbar", "errorband"}

    @classmethod
    def apply(cls, spec: dict[str, Any], *, data: pd.DataFrame | None) -> AxisDomainPolicyResult:
        changes: list[str] = []
        for unit in iter_unit_specs(spec):
            mark_type = cls._mark_type(unit)
            encoding = unit.get("encoding")
            if not isinstance(encoding, dict):
                continue
            for channel in cls.POSITION_CHANNELS:
                channel_def = encoding.get(channel)
                if not isinstance(channel_def, dict):
                    continue
                if not cls._is_quantitative(channel_def):
                    continue
                decision = cls._zero_decision(channel_def, mark_type=mark_type, data=data)
                if decision is None:
                    continue
                scale = channel_def.setdefault("scale", {})
                if not isinstance(scale, dict):
                    continue
                previous = scale.get("zero")
                if previous != decision:
                    scale["zero"] = decision
                    changes.append(f"Set encoding.{channel}.scale.zero={decision} for {mark_type or 'unit'} view.")
        return AxisDomainPolicyResult(changes=changes)

    @classmethod
    def _zero_decision(cls, channel_def: dict[str, Any], *, mark_type: str, data: pd.DataFrame | None) -> bool | None:
        field = channel_def.get("field") if isinstance(channel_def.get("field"), str) else ""
        aggregate = str(channel_def.get("aggregate") or "").strip().lower()
        field_key = field.strip().lower()

        if aggregate in cls.ZERO_BASELINE_AGGREGATES or field_key in cls.ZERO_BASELINE_FIELD_TOKENS:
            return True
        if any(token in field_key for token in cls.ZERO_BASELINE_FIELD_TOKENS):
            return True
        if any(token in field_key for token in cls.PROPORTION_FIELD_TOKENS) and mark_type in cls.BASELINE_MARKS:
            return True
        if channel_def.get("bin"):
            return False
        if mark_type in cls.DISTRIBUTION_MARKS:
            return False
        if mark_type in cls.BASELINE_MARKS:
            return True
        if cls._zero_wastes_domain(field, data):
            return False
        return None

    @staticmethod
    def _zero_wastes_domain(field: str, data: pd.DataFrame | None) -> bool:
        if not field or data is None or field not in data.columns:
            return False
        series = pd.to_numeric(data[field], errors="coerce").dropna()
        if series.empty:
            return False
        min_value = float(series.min())
        max_value = float(series.max())
        if not (isfinite(min_value) and isfinite(max_value)):
            return False
        if min_value == max_value:
            return min_value != 0.0
        span = abs(max_value - min_value)
        magnitude = max(abs(min_value), abs(max_value), 1e-12)
        same_positive_side = min_value > 0 and max_value > 0
        same_negative_side = min_value < 0 and max_value < 0
        return bool((same_positive_side or same_negative_side) and span / magnitude < 0.65)

    @staticmethod
    def _mark_type(unit: dict[str, Any]) -> str:
        mark = unit.get("mark")
        if isinstance(mark, str):
            return mark.strip().lower()
        if isinstance(mark, dict):
            value = mark.get("type")
            if isinstance(value, str):
                return value.strip().lower()
        return ""

    @staticmethod
    def _is_quantitative(channel_def: dict[str, Any]) -> bool:
        type_name = channel_def.get("type")
        if isinstance(type_name, str) and type_name.strip().lower() == "quantitative":
            return True
        aggregate = channel_def.get("aggregate")
        return isinstance(aggregate, str) and aggregate.strip().lower() == "count"
