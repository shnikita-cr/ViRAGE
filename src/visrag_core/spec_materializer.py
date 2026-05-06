from __future__ import annotations

from copy import deepcopy
from typing import Any

from .chart_types import canonicalize_chart_type, normalize_aggregate
from .models import VisRAGExample
from .semantic import vega_type

_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"


def materialize_spec_template(example: VisRAGExample, field_mapping: dict[str, str]) -> dict[str, Any]:
    spec = deepcopy(example.spec_template)
    if not isinstance(spec, dict) or not spec:
        return {}
    spec["$schema"] = spec.get("$schema") or _SCHEMA
    spec["mark"] = _normalize_mark(spec.get("mark") or example.chart_type)
    encoding = spec.setdefault("encoding", {})
    if not isinstance(encoding, dict):
        return spec
    for channel, field_name in field_mapping.items():
        role = example.field_roles.get(channel, "nominal")
        channel_spec = encoding.get(channel)
        if not isinstance(channel_spec, dict):
            channel_spec = {}
            encoding[channel] = channel_spec
        channel_spec["field"] = field_name
        channel_spec.setdefault("type", vega_type(role))
        aggregate = normalize_aggregate(channel_spec.get("aggregate"))
        if aggregate:
            channel_spec["aggregate"] = aggregate
    _normalize_aggregates(encoding)
    return spec


def _normalize_mark(mark: Any) -> Any:
    if isinstance(mark, dict):
        clone = dict(mark)
        clone["type"] = canonicalize_chart_type(clone.get("type"))
        return clone
    return canonicalize_chart_type(mark)


def _normalize_aggregates(encoding: dict[str, Any]) -> None:
    for channel_spec in encoding.values():
        if isinstance(channel_spec, dict):
            aggregate = normalize_aggregate(channel_spec.get("aggregate"))
            if aggregate:
                channel_spec["aggregate"] = aggregate
        elif isinstance(channel_spec, list):
            for item in channel_spec:
                if isinstance(item, dict):
                    aggregate = normalize_aggregate(item.get("aggregate"))
                    if aggregate:
                        item["aggregate"] = aggregate
