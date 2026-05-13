from __future__ import annotations

import json
from typing import Any, Iterable

from src.services.spec_metrics.models import COMPOSITION_KEYS, EncodingItem, FACET_EQUIVALENT, TransformItem, View
from src.visrag_core import normalize_aggregate


def extract_views(spec: Any) -> list[View]:
    if not isinstance(spec, dict):
        return []
    views: list[View] = []
    _collect_views(spec, inherited_transforms=(), views=views)
    return views or [View(mark="", encodings=(), transforms=tuple(_extract_transforms(spec)))]


def all_encodings(views: Iterable[View]) -> list[EncodingItem]:
    return [encoding for view in views for encoding in view.encodings]


def all_transforms(views: Iterable[View]) -> list[TransformItem]:
    return [transform for view in views for transform in view.transforms]


def _collect_views(node: Any, *, inherited_transforms: tuple[TransformItem, ...], views: list[View]) -> None:
    if not isinstance(node, dict):
        return
    local_transforms = (*inherited_transforms, *_extract_transforms(node))
    if "mark" in node or "encoding" in node:
        encodings = tuple(_extract_encodings(node.get("encoding")))
        views.append(View(
            mark=_mark_value(node.get("mark")),
            encodings=encodings,
            transforms=(*local_transforms, *_inline_transform_items(encodings)),
        ))
    for key in COMPOSITION_KEYS:
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                _collect_views(child, inherited_transforms=local_transforms, views=views)
    for key in ("spec", "child"):
        child = node.get(key)
        if isinstance(child, dict):
            _collect_views(child, inherited_transforms=local_transforms, views=views)
    _collect_faceted_view(node, local_transforms, views)


def _collect_faceted_view(node: dict[str, Any], local_transforms: tuple[TransformItem, ...], views: list[View]) -> None:
    facet = node.get("facet")
    if not isinstance(facet, dict) or "spec" not in node:
        return
    facet_encodings = tuple(_extract_facet_encodings(facet))
    child_before = len(views)
    _collect_views(node["spec"], inherited_transforms=local_transforms, views=views)
    for index in range(child_before, len(views)):
        view = views[index]
        views[index] = View(mark=view.mark, encodings=(*view.encodings, *facet_encodings), transforms=view.transforms)


def _mark_value(mark: Any) -> str:
    return str(mark.get("type") or "") if isinstance(mark, dict) else str(mark or "")


def _extract_encodings(encoding: Any) -> list[EncodingItem]:
    if not isinstance(encoding, dict):
        return []
    result: list[EncodingItem] = []
    for channel, channel_spec in encoding.items():
        for field_def in _iter_field_defs(channel_spec):
            item = _encoding_item(str(channel), field_def)
            if item.field or item.aggregate == "count":
                result.append(item)
    return result


def _extract_facet_encodings(facet: dict[str, Any]) -> list[EncodingItem]:
    result: list[EncodingItem] = []
    for channel in ("row", "column", "facet"):
        value = facet.get(channel) if channel in facet else (facet if "field" in facet else None)
        if isinstance(value, dict):
            result.append(_encoding_item(channel, value))
    return result


def _iter_field_defs(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if "field" in value or value.get("aggregate") == "count":
            yield value
        nested = value.get("condition")
        if isinstance(nested, dict):
            yield from _iter_field_defs(nested)
        elif isinstance(nested, list):
            for item in nested:
                yield from _iter_field_defs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_field_defs(item)


def _encoding_item(channel: str, channel_spec: dict[str, Any]) -> EncodingItem:
    aggregate = str(normalize_aggregate(channel_spec.get("aggregate")) or "").strip().lower()
    field = "*" if aggregate == "count" and not channel_spec.get("field") else str(channel_spec.get("field") or "")
    return EncodingItem(
        channel=_normalize_channel(channel),
        field=field.strip(),
        type=str(channel_spec.get("type") or "").strip().lower(),
        aggregate=aggregate,
        bin=_normalize_flag(channel_spec.get("bin")),
        time_unit=str(channel_spec.get("timeUnit") or channel_spec.get("timeunit") or "").strip().lower(),
    )


def _normalize_channel(channel: str) -> str:
    lowered = channel.strip().lower()
    return "facet" if lowered in FACET_EQUIVALENT else lowered


def _normalize_flag(value: Any) -> str:
    if value is None or value is False:
        return ""
    if value is True:
        return "true"
    return stable_json(value)


def _inline_transform_items(encodings: Iterable[EncodingItem]) -> list[TransformItem]:
    result: list[TransformItem] = []
    for item in encodings:
        if item.aggregate:
            result.append(TransformItem("aggregate", item.field, item.aggregate, item.field, (), ""))
        if item.bin:
            result.append(TransformItem("bin", item.field, item.bin, item.field, (), ""))
        if item.time_unit:
            result.append(TransformItem("timeUnit", item.field, item.time_unit, item.field, (), ""))
    return result


def _extract_transforms(node: Any) -> list[TransformItem]:
    transforms = node.get("transform") if isinstance(node, dict) else None
    if not isinstance(transforms, list):
        return []
    result: list[TransformItem] = []
    for transform in transforms:
        if isinstance(transform, dict):
            result.extend(_transform_items_from_dict(transform))
    return result


def _transform_items_from_dict(transform: dict[str, Any]) -> list[TransformItem]:
    if "aggregate" in transform or "joinaggregate" in transform:
        return _aggregate_transform_items(transform, "aggregate" if "aggregate" in transform else "joinaggregate")
    for key in ("bin", "timeUnit", "timeunit"):
        if key in transform:
            kind = "timeUnit" if key.lower() == "timeunit" else key
            return [TransformItem(kind, str(transform.get("field") or ""), stable_json(transform.get(key)),
                                  str(transform.get("as") or ""), (), "")]
    for key in ("filter", "calculate", "lookup", "window", "fold", "pivot", "impute", "density", "loess", "regression"):
        if key in transform:
            return [TransformItem(key, _first_field(transform), "", str(transform.get("as") or ""),
                                  _tuple_strings(transform.get("groupby")), stable_json(transform))]
    return [TransformItem("unknown", _first_field(transform), "", str(transform.get("as") or ""), (),
                          stable_json(transform))]


def _aggregate_transform_items(transform: dict[str, Any], key: str) -> list[TransformItem]:
    result: list[TransformItem] = []
    for item in transform.get(key) or []:
        if isinstance(item, dict):
            result.append(TransformItem(key, str(item.get("field") or "*"), str(item.get("op") or ""),
                                        str(item.get("as") or ""), _tuple_strings(transform.get("groupby")), ""))
    return result


def _tuple_strings(value: Any) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in value if isinstance(item, str))) if isinstance(value, list) else ()


def _first_field(value: Any) -> str:
    if isinstance(value, dict):
        field = value.get("field")
        if isinstance(field, str):
            return field
        for item in value.values():
            found = _first_field(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _first_field(item)
            if found:
                return found
    return ""


def stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)
