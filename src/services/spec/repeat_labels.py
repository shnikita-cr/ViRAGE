from __future__ import annotations

import re
from typing import Any

GENERIC_REPEAT_TITLES = {
    "",
    "value",
    "values",
    "metric",
    "metrics",
    "measure",
    "measurement",
    "repeated metric",
    "repeated metrics",
    "repeated metric value",
    "repeated metrics value",
    "значение",
}

_UPPERCASE_TOKENS = {"psnr", "ssim", "lpips", "mse", "rmse", "mae", "ms-ssim", "uiqi", "vif", "fsim", "nlv"}
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def repeat_fields(spec: dict[str, Any]) -> list[str]:
    repeat = spec.get("repeat")
    fields: list[str] = []
    if isinstance(repeat, dict):
        for value in repeat.values():
            fields.extend(_repeat_value_fields(value))
    else:
        fields.extend(_repeat_value_fields(repeat))
    return _dedupe(fields)


def repeat_header_label_expression() -> str:
    return "replace(datum.label, /_/g, ' ')"


def ensure_repeat_header_config(spec: dict[str, Any]) -> bool:
    if not repeat_fields(spec):
        return False
    config = spec.setdefault("config", {})
    if not isinstance(config, dict):
        return False
    header = config.setdefault("header", {})
    if not isinstance(header, dict):
        return False
    changed = False
    for key, value in {
        "labelExpr": repeat_header_label_expression(),
        "labelFontSize": 12,
        "titleFontSize": 12,
        "labelLimit": 220,
        "titleLimit": 260,
    }.items():
        if key not in header:
            header[key] = value
            changed = True
    return changed


def repeat_field_title(fields: list[str]) -> str:
    labels = [_humanize(field) for field in fields if str(field).strip()]
    if not labels:
        return "Panel metric"
    return _join_labels(labels, limit=4)


def repeat_axis_title(*, aggregate: str | None = None, value_role: str = "value") -> str:
    prefix = _aggregate_label(aggregate)
    role = value_role.strip() or "value"
    return f"{prefix}Panel metric {role}" if prefix else f"Panel metric {role}"


def repeat_chart_title(*, repeated_fields: list[str], group_labels: list[str], mark_type: str, aggregate: str | None) -> str:
    field_title = repeat_field_title(repeated_fields)
    group_suffix = f" by {_join_labels(group_labels, limit=3)}" if group_labels else ""
    if mark_type in {"boxplot", "errorband", "errorbar"}:
        return f"{field_title} Distributions{group_suffix}"
    aggregate_label = _aggregate_label(aggregate).strip()
    metric_text = f"{aggregate_label} {field_title}" if aggregate_label else field_title
    return f"{metric_text}{group_suffix}"


def title_is_generic_repeat(value: Any) -> bool:
    text = title_text(value).strip().lower()
    return text in GENERIC_REPEAT_TITLES or text.startswith("repeated metrics:")


def title_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text.strip()
        if isinstance(text, list):
            return " ".join(str(item).strip() for item in text if str(item).strip())
    return ""


def _repeat_value_fields(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _humanize(value: str) -> str:
    words = _WORD_RE.findall(str(value).replace("_", " ").replace("-", " "))
    if not words:
        return str(value).strip()
    return " ".join(_word_label(word) for word in words)


def _word_label(word: str) -> str:
    lower = word.lower()
    if lower in _UPPERCASE_TOKENS:
        return lower.upper()
    return word[:1].upper() + word[1:]


def _join_labels(labels: list[str], *, limit: int) -> str:
    values = [label for label in labels if label]
    if len(values) > limit:
        values = [*values[:limit], f"{len(labels) - limit} more"]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])} and {values[-1]}"


def _aggregate_label(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    labels = {
        "mean": "Average ",
        "average": "Average ",
        "sum": "Total ",
        "count": "Count ",
        "median": "Median ",
        "min": "Minimum ",
        "max": "Maximum ",
    }
    return labels.get(normalized, "")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result
