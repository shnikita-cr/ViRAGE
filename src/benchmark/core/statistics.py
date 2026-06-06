from __future__ import annotations

from collections.abc import Sequence


def mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def mean_bool(values: Sequence[bool]) -> float | None:
    return round(sum(1 for value in values if value) / len(values), 6) if values else None


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 6)
    return round((ordered[middle - 1] + ordered[middle]) / 2.0, 6)


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 6)
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile_value))))
    return round(ordered[index], 6)
