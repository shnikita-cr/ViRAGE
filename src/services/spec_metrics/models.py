from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MARK_ALIASES = {
    "scatter": "point",
    "symbol": "point",
    "circle": "point",
    "square": "point",
}
PARTIAL_MARK_GROUPS = [
    {"point", "circle", "square", "symbol"},
    {"line", "area", "trail"},
    {"bar", "rect", "tick"},
    {"arc", "pie", "donut"},
]
MARK_WORDS = {
    "bar", "line", "area", "point", "scatter", "circle", "square", "tick", "rect", "arc", "pie", "donut",
    "text", "rule", "boxplot", "histogram", "heatmap",
}
POSITIONAL_SWAP = {"x": "y", "y": "x"}
FACET_EQUIVALENT = {"row", "column", "facet"}
COMPOSITION_KEYS = ("layer", "concat", "hconcat", "vconcat")


@dataclass(frozen=True)
class View:
    mark: str
    encodings: tuple["EncodingItem", ...]
    transforms: tuple["TransformItem", ...]


@dataclass(frozen=True)
class EncodingItem:
    channel: str
    field: str
    type: str
    aggregate: str
    bin: str
    time_unit: str


@dataclass(frozen=True)
class TransformItem:
    kind: str
    field: str
    op: str
    as_field: str
    groupby: tuple[str, ...]
    signature: str


@dataclass(frozen=True)
class MatchStats:
    score: float
    precision: float
    recall: float
    matches: list[tuple[Any, Any, float]]
