from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


POSITION_CHANNELS = {"x", "y", "x2", "y2"}
LEGEND_CHANNELS = {"color", "shape", "fill", "stroke", "strokeDash", "opacity", "size"}
DISCRETE_TYPES = {"nominal", "ordinal"}
QUANTITATIVE_TYPES = {"quantitative"}
COMPOSITION_KEYS = ("layer", "hconcat", "vconcat", "concat", "spec")


def clone_jsonish(value: Any) -> Any:
    return deepcopy(value)


def walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_dicts(item)


def iter_unit_specs(spec: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(spec, dict):
        return
    if isinstance(spec.get("encoding"), dict) or "mark" in spec:
        yield spec
    for key in COMPOSITION_KEYS:
        child = spec.get(key)
        if isinstance(child, list):
            for item in child:
                yield from iter_unit_specs(item)
        elif isinstance(child, dict):
            yield from iter_unit_specs(child)


def mark_type(spec: dict[str, Any]) -> str:
    mark = spec.get("mark")
    if isinstance(mark, str):
        return mark.strip().lower()
    if isinstance(mark, dict):
        return str(mark.get("type") or "").strip().lower()
    return ""


def channel_field(channel_def: Any) -> str | None:
    if isinstance(channel_def, dict):
        field = channel_def.get("field")
        if isinstance(field, str) and field.strip():
            return field.strip()
    return None


def channel_type(channel_def: Any) -> str | None:
    if isinstance(channel_def, dict):
        type_name = channel_def.get("type")
        if isinstance(type_name, str) and type_name.strip():
            return type_name.strip().lower()
    return None


def is_discrete_channel(channel_def: Any) -> bool:
    return channel_type(channel_def) in DISCRETE_TYPES


def is_quantitative_channel(channel_def: Any) -> bool:
    return channel_type(channel_def) in QUANTITATIVE_TYPES


def get_encoding(unit: dict[str, Any]) -> dict[str, Any]:
    encoding = unit.get("encoding")
    return encoding if isinstance(encoding, dict) else {}


def set_axis_property(channel_def: dict[str, Any], key: str, value: Any, *, overwrite: bool = False) -> bool:
    axis = channel_def.get("axis")
    if axis is None:
        axis = {}
        channel_def["axis"] = axis
    if axis is False or not isinstance(axis, dict):
        return False
    if overwrite or key not in axis:
        axis[key] = value
        return True
    return False


def set_scale_property(channel_def: dict[str, Any], key: str, value: Any, *, overwrite: bool = False) -> bool:
    scale = channel_def.get("scale")
    if scale is None:
        scale = {}
        channel_def["scale"] = scale
    if scale is False or not isinstance(scale, dict):
        return False
    if overwrite or key not in scale:
        scale[key] = value
        return True
    return False


def legend_is_visible(channel_def: dict[str, Any]) -> bool:
    legend = channel_def.get("legend")
    return legend is not None and legend is not False


def field_values(data: Any, field: str | None) -> list[Any]:
    if field is None or data is None or field not in getattr(data, "columns", []):
        return []
    series = data[field].dropna()
    return list(series)


def numeric_range(data: Any, field: str | None) -> tuple[float | None, float | None]:
    if field is None or data is None or field not in getattr(data, "columns", []):
        return None, None
    try:
        series = data[field].dropna().astype(float)
    except (TypeError, ValueError):
        return None, None
    if series.empty:
        return None, None
    return float(series.min()), float(series.max())
