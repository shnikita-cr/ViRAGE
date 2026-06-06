from __future__ import annotations

from typing import Any

_TITLE_KEYS = ("title",)


def chart_text_consistency_score(spec: dict[str, Any]) -> float | None:
    """Return 1.0 when labels for the same encoded metric are internally consistent.

    The check is deliberately structural: it groups channel labels by the same
    field + aggregate/bin/timeUnit identity. It does not maintain a synonym
    dictionary and does not judge semantic quality of the text.
    """
    if not isinstance(spec, dict):
        return None
    labels_by_metric = _collect_labels_by_metric(spec)
    if not labels_by_metric:
        return None
    inconsistent = 0
    checked = 0
    for labels in labels_by_metric.values():
        normalized = {_normalize_label(label) for label in labels if str(label).strip()}
        if len(normalized) <= 1:
            continue
        checked += 1
        inconsistent += 1
    if checked == 0:
        return 1.0
    return 0.0 if inconsistent else 1.0


def _collect_labels_by_metric(spec: dict[str, Any]) -> dict[tuple[str, str, str, str], list[str]]:
    labels_by_metric: dict[tuple[str, str, str, str], list[str]] = {}
    for encoding in _iter_encoding_blocks(spec):
        for channel, channel_def in encoding.items():
            if isinstance(channel_def, list):
                for item in channel_def:
                    _collect_channel_labels(labels_by_metric, channel, item)
            else:
                _collect_channel_labels(labels_by_metric, channel, channel_def)
    return labels_by_metric


def _iter_encoding_blocks(value: Any):
    if isinstance(value, dict):
        encoding = value.get("encoding")
        if isinstance(encoding, dict):
            yield encoding
        for child in value.values():
            yield from _iter_encoding_blocks(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_encoding_blocks(item)


def _collect_channel_labels(
        labels_by_metric: dict[tuple[str, str, str, str], list[str]],
        channel: str,
        channel_def: Any,
) -> None:
    if not isinstance(channel_def, dict):
        return
    field = str(channel_def.get("field") or "").strip()
    if not field:
        return
    aggregate = str(channel_def.get("aggregate") or "").strip()
    time_unit = str(channel_def.get("timeUnit") or "").strip()
    bin_value = "bin" if channel_def.get("bin") else ""
    metric_key = (field, aggregate, time_unit, bin_value)
    labels = labels_by_metric.setdefault(metric_key, [])
    direct_title = channel_def.get("title")
    if isinstance(direct_title, str) and direct_title.strip():
        labels.append(direct_title)
    axis = channel_def.get("axis")
    if isinstance(axis, dict):
        axis_title = axis.get("title")
        if isinstance(axis_title, str) and axis_title.strip():
            labels.append(axis_title)
    legend = channel_def.get("legend")
    if isinstance(legend, dict):
        legend_title = legend.get("title")
        if isinstance(legend_title, str) and legend_title.strip():
            labels.append(legend_title)


def _normalize_label(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())
