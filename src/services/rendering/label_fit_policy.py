from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.services.rendering.common.chart_render_policy_types import ChartRenderPolicyResult
from src.services.rendering.common.spec_traversal import iter_unit_specs


@dataclass(frozen=True)
class LabelFitPolicyResult:
    padding: dict[str, int]
    changes: list[str] = field(default_factory=list)


class LabelFitPolicy:
    """Protect axis labels from clipping while keeping the logical canvas compact."""

    DISCRETE_TYPES = {"nominal", "ordinal"}

    @classmethod
    def apply(
            cls,
            spec: dict[str, Any],
            *,
            data: pd.DataFrame | None,
            render_policy: ChartRenderPolicyResult,
    ) -> LabelFitPolicyResult:
        padding = {"left": 20, "right": 16, "top": 18, "bottom": 24}
        changes: list[str] = []
        for unit in iter_unit_specs(spec):
            encoding = unit.get("encoding")
            if not isinstance(encoding, dict):
                continue
            for channel in ("x", "y"):
                channel_def = encoding.get(channel)
                if not isinstance(channel_def, dict):
                    continue
                axis = channel_def.setdefault("axis", {})
                if not isinstance(axis, dict):
                    continue
                field = channel_def.get("field") if isinstance(channel_def.get("field"), str) else None
                cardinality, longest = cls._field_cardinality_and_label(field, data)
                label_limit = cls._label_limit(longest)
                if int(axis.get("labelLimit") or 0) < label_limit:
                    axis["labelLimit"] = label_limit
                    changes.append(f"Set encoding.{channel}.axis.labelLimit={label_limit}.")
                axis.setdefault("labelBound", True)
                axis.setdefault("labelOverlap", "greedy")

                if channel == "x" and cls._is_discrete(channel_def, field, data):
                    slot_width = max(1.0, render_policy.width / max(1, cardinality))
                    label_angle = cls._x_label_angle(longest=longest, slot_width=slot_width)
                    if label_angle is not None and "labelAngle" not in axis:
                        axis["labelAngle"] = label_angle
                        changes.append(f"Set encoding.x.axis.labelAngle={label_angle} for crowded category labels.")
                    angle = int(axis.get("labelAngle") or 0)
                    if abs(angle) >= 80:
                        padding["bottom"] = max(padding["bottom"], min(150, 40 + longest * 5))
                    elif abs(angle) > 0:
                        padding["bottom"] = max(padding["bottom"], min(112, 32 + longest * 3))
                    else:
                        padding["bottom"] = max(padding["bottom"], 40 if longest > 10 else 28)
                if channel == "y" and cls._is_discrete(channel_def, field, data):
                    padding["left"] = max(padding["left"], min(220, 44 + longest * 6))
                if channel == "y" and cls._is_count_like(channel_def):
                    axis.setdefault("tickMinStep", 1)
                    changes.append("Set encoding.y.axis.tickMinStep=1 for count-like quantitative axis.")
        return LabelFitPolicyResult(padding=padding, changes=changes)

    @staticmethod
    def _x_label_angle(*, longest: int, slot_width: float) -> int | None:
        estimated_label_width = max(1, longest) * 6.4
        if estimated_label_width <= slot_width * 0.95:
            return None
        if estimated_label_width <= slot_width * 1.45:
            return -35
        return -90

    @classmethod
    def _is_discrete(cls, channel_def: dict[str, Any], field: str | None, data: pd.DataFrame | None) -> bool:
        type_name = channel_def.get("type")
        if isinstance(type_name, str) and type_name.strip().lower() in cls.DISCRETE_TYPES:
            return True
        if not field or data is None or field not in data.columns:
            return False
        dtype = data[field].dtype
        return bool(
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        )

    @staticmethod
    def _field_cardinality_and_label(field: str | None, data: pd.DataFrame | None) -> tuple[int, int]:
        if not field or data is None or field not in data.columns:
            return 1, len(str(field or ""))
        series = data[field].dropna()
        if series.empty:
            return 1, len(str(field))
        values = series.astype(str)
        return max(1, int(values.nunique(dropna=True))), max(len(str(field)), int(values.map(len).max()))

    @staticmethod
    def _label_limit(longest: int) -> int:
        return max(140, min(420, 12 * max(1, int(longest))))

    @staticmethod
    def _is_count_like(channel_def: dict[str, Any]) -> bool:
        aggregate = str(channel_def.get("aggregate") or "").strip().lower()
        field = str(channel_def.get("field") or "").strip().lower()
        return aggregate == "count" or field in {"count", "frequency", "records", "missing_count"}
