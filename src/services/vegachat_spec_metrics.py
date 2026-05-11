from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

SpecType = dict[str, Any]

VEGA_LITE_MARKS = [
    "bar",
    "circle",
    "square",
    "tick",
    "line",
    "area",
    "point",
    "geoshape",
    "rule",
    "text",
    "boxplot",
    "errorband",
    "errorbar",
]

VEGA_LITE_MARKS_SYNONYMS = [
    "donut",
    "map",
    "pie",
    "heatmap",
    "dot",
    "array",
    "scatterplot",
    "scatter",
    "histogram",
    "column",
    "row",
    "cluster",
]


class ListIndex(NamedTuple):
    i: int


class F1Score(NamedTuple):
    f1: float
    precision: float
    recall: float


@dataclass(frozen=True)
class VegaChatSpecScoreResult:
    spec_score: float
    mark: F1Score
    encoding: F1Score
    transform: F1Score
    full: F1Score
    keys: F1Score
    kvs: F1Score
    keys_jaccard: float
    is_drawable: bool
    is_empty_chart: bool
    is_valid_schema: bool
    details: list[str]

    def to_metrics_dict(self) -> dict[str, float]:
        return {
            "spec_score": self.spec_score,
            **{f"mark_{k}": v for k, v in self.mark._asdict().items()},
            **{f"encoding_{k}": v for k, v in self.encoding._asdict().items()},
            **{f"transform_{k}": v for k, v in self.transform._asdict().items()},
            **{f"full_{k}": v for k, v in self.full._asdict().items()},
            **{f"keys_{k}": v for k, v in self.keys._asdict().items()},
            **{f"kvs_{k}": v for k, v in self.kvs._asdict().items()},
            "keys_jaccard": self.keys_jaccard,
            "is_drawable": float(self.is_drawable),
            "is_empty_chart": float(self.is_empty_chart),
            "is_valid_schema": float(self.is_valid_schema),
        }


def f_beta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision + recall <= 0.0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)


def compute_f1(list_ref: list[Any], list_hyp: list[Any], beta: float = 1.0) -> F1Score:
    ref = Counter(list_ref)
    hyp = Counter(list_hyp)
    true = sum(ref.values())
    positive = sum(hyp.values())
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    return F1Score(f_beta_score(precision, recall, beta=beta), precision, recall)


def compute_f1_weighted(
    list_ref: list[tuple[Any, float]], list_hyp: list[tuple[Any, float]], beta: float = 1.0
) -> F1Score:
    ref: Counter[Any] = Counter()
    hyp: Counter[Any] = Counter()
    for item, weight in list_ref:
        if weight > 0:
            ref[item] += weight
    for item, weight in list_hyp:
        if weight > 0:
            hyp[item] += weight
    true = sum(ref.values())
    positive = sum(hyp.values())
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    return F1Score(f_beta_score(precision, recall, beta=beta), precision, recall)


def get_spec_paths(value: Any) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    if isinstance(value, dict):
        if len(value) == 0:
            out.append(tuple())
        for key, child in value.items():
            out.extend((key, *path) for path in get_spec_paths(child))
    elif isinstance(value, list):
        if len(value) == 0:
            out.append(tuple())
        for index, child in enumerate(value):
            out.extend((ListIndex(index), *path) for path in get_spec_paths(child))
    else:
        out.append((value,))
    return out


def spec_paths_ignore_list_order(paths: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [tuple(ListIndex(0) if isinstance(item, ListIndex) else item for item in path) for path in paths]


def get_spec_keys(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            out.append(str(key))
            out.extend(get_spec_keys(child))
    elif isinstance(value, list):
        for child in value:
            out.extend(get_spec_keys(child))
    return out


def get_spec_leaf_key_values(spec: SpecType) -> list[tuple[Any, ...]]:
    paths = get_spec_paths(spec)
    paths = [tuple(ListIndex(0) if isinstance(item, ListIndex) else item for item in path) for path in paths]
    kvs: list[tuple[Any, ...]] = []
    for path in paths:
        index = len(path) - 2
        while index >= 0 and isinstance(path[index], ListIndex):
            index -= 1
        if index >= 0 and not isinstance(path[index], ListIndex):
            kvs.append(path[index:])
    return kvs


def get_spec_field(value: Any, field: str) -> list[Any]:
    out: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == field:
                out.append(child)
            else:
                out.extend(get_spec_field(child, field))
    elif isinstance(value, list):
        for child in value:
            out.extend(get_spec_field(child, field))
    return out


def get_spec_marks(spec: SpecType) -> list[str]:
    marks: list[str] = []
    for mark in get_spec_field(spec, "mark"):
        if isinstance(mark, str):
            marks.append(mark)
        elif isinstance(mark, dict) and "type" in mark:
            marks.append(str(mark["type"]))
    return marks


def get_spec_transform_paths(spec: SpecType) -> list[tuple[Any, ...]]:
    paths: list[tuple[Any, ...]] = []
    for transform in get_spec_field(spec, "transform"):
        transform_paths = get_spec_paths(transform)
        transform_paths = [p for p in transform_paths if len(p) > 0 and not isinstance(p[-1], ListIndex)]
        paths.extend(transform_paths)
    return paths


def spec_f1_correctness_mark(spec_ref: SpecType, spec_hyp: SpecType) -> F1Score:
    def apply_mark_equivalence(marks: list[str]) -> list[str]:
        out: list[str] = []
        for mark in marks:
            mark = str(mark)
            if mark in ("circle", "point", "square"):
                out.append("circle-point-square")
            else:
                out.append(mark)
        return out

    hyp_marks = get_spec_marks(spec_hyp)
    ref_marks = get_spec_marks(spec_ref)
    original = compute_f1(ref_marks, hyp_marks)
    equivalent = compute_f1(apply_mark_equivalence(ref_marks), apply_mark_equivalence(hyp_marks))
    return F1Score(
        f1=(original.f1 + equivalent.f1) / 2.0,
        precision=(original.precision + equivalent.precision) / 2.0,
        recall=(original.recall + equivalent.recall) / 2.0,
    )


def get_my_encoding_fields(spec: SpecType, *, include_titles: bool = False) -> list[tuple[str, str, Any]]:
    channels = (
        ["x", "y", "x2", "y2", "xError", "yError", "xError2", "yError2"]
        + ["color", "row", "column"]
        + ["theta", "theta2", "radius", "radius2"]
        + ["longitude", "latitude", "longitude2", "latitude2"]
    )
    properties = ["field", "type", "aggregate", "bin", "timeUnit"] + (["title", "axis"] if include_titles else [])
    values: list[tuple[str, str, Any]] = []
    for encoding in get_spec_field(spec, "encoding"):
        if not isinstance(encoding, dict):
            continue
        for channel in channels:
            if channel not in encoding or not isinstance(encoding[channel], dict):
                continue
            channel_spec = encoding[channel]
            for prop in properties:
                if prop not in channel_spec:
                    continue
                value = channel_spec[prop]
                prop_out = prop
                if prop == "field" and isinstance(value, dict):
                    if "repeat" in value:
                        value = value["repeat"]
                    else:
                        continue
                elif prop == "bin" and isinstance(value, dict):
                    continue
                elif prop == "title" and isinstance(value, list):
                    value = "".join(str(item) for item in value)
                elif prop == "axis" and isinstance(value, dict):
                    if "title" not in value:
                        continue
                    value = value["title"]
                    prop_out = "title"
                values.append((channel, prop_out, _normalize_value(value)))
    return values


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def spec_f1_correctness_encoding(
    spec_ref: SpecType,
    spec_hyp: SpecType,
    *,
    swappable_xy: bool = True,
    swappable_faceting: bool = True,
    include_titles: bool = False,
    types_weight: float = 0.5,
    time_unit_weight: float = 0.5,
    beta: float = 2.0,
) -> F1Score:
    def do_swappable_fields(values: list[tuple[str, str, Any]]) -> list[tuple[str, str, Any]]:
        out: list[tuple[str, str, Any]] = []
        for channel, field, value in values:
            channel_out = channel
            if swappable_xy and channel[:1] in ("x", "y"):
                channel_out = "xy" + channel[1:]
            elif swappable_faceting and channel in ("row", "column"):
                channel_out = "row|column"
            out.append((channel_out, field, value))
        return out

    def apply_weights(values: list[tuple[str, str, Any]]) -> list[tuple[tuple[str, str, Any], float]]:
        out: list[tuple[tuple[str, str, Any], float]] = []
        for channel, field, value in values:
            weight = 1.0
            if field == "type":
                weight = types_weight
            elif field == "timeUnit":
                weight = time_unit_weight
            out.append(((channel, field, value), weight))
        return out

    hyp_values = apply_weights(do_swappable_fields(get_my_encoding_fields(spec_hyp, include_titles=include_titles)))
    ref_values = apply_weights(do_swappable_fields(get_my_encoding_fields(spec_ref, include_titles=include_titles)))
    return compute_f1_weighted(ref_values, hyp_values, beta=beta)


def spec_f1_correctness_transform(spec_ref: SpecType, spec_hyp: SpecType) -> F1Score:
    ref_paths = spec_paths_ignore_list_order(get_spec_transform_paths(spec_ref))
    hyp_paths = spec_paths_ignore_list_order(get_spec_transform_paths(spec_hyp))
    return compute_f1(ref_paths, hyp_paths)


def spec_f1_correctness_full(spec_ref: SpecType, spec_hyp: SpecType) -> F1Score:
    ref_paths = spec_paths_ignore_list_order(get_spec_paths(spec_ref))
    hyp_paths = spec_paths_ignore_list_order(get_spec_paths(spec_hyp))
    return compute_f1(ref_paths, hyp_paths)


def spec_f1_correctness_key_values(spec_ref: SpecType, spec_hyp: SpecType) -> F1Score:
    return compute_f1(get_spec_leaf_key_values(spec_ref), get_spec_leaf_key_values(spec_hyp))


def spec_f1_correctness_key(spec_ref: SpecType, spec_hyp: SpecType) -> F1Score:
    return compute_f1(get_spec_keys(spec_ref), get_spec_keys(spec_hyp))


def jaccard_similarity(a: set[Any], b: set[Any]) -> float:
    union = len(a | b)
    if union == 0:
        return float("nan")
    return len(a & b) / union


def spec_jaccard_keys(spec_ref: SpecType, spec_hyp: SpecType) -> float:
    return jaccard_similarity(set(get_spec_keys(spec_ref)), set(get_spec_keys(spec_hyp)))


def get_marks_in_utterance(utterance: str) -> list[str]:
    mark_keywords = set(VEGA_LITE_MARKS + VEGA_LITE_MARKS_SYNONYMS)
    return [word.strip() for word in utterance.lower().split() if word.strip() in mark_keywords]


def _prompt_mentions_mark_fallback(utterance: str) -> bool:
    mark_keywords = set(VEGA_LITE_MARKS + VEGA_LITE_MARKS_SYNONYMS)
    tokens = set(re.findall(r"[a-zA-Z]+", utterance.lower()))
    return bool(tokens & mark_keywords)


def spec_score_impl(
    spec_ref: SpecType,
    spec_hyp: SpecType,
    *,
    utterance: str,
    hyp_is_drawable: bool,
    hyp_is_empty_chart: bool,
    hyp_is_valid_schema: bool,
) -> float:
    if not hyp_is_drawable:
        return 0.0

    w_drawable = 0.005
    x_drawable = w_drawable

    w_valid_schema = w_drawable if hyp_is_valid_schema else 1.0
    x_valid_schema = w_valid_schema if hyp_is_valid_schema else 0.0

    w_not_empty = 1000.0 if hyp_is_empty_chart else w_drawable
    x_not_empty = 0.0 if hyp_is_empty_chart else w_not_empty

    w_encoding_penalty = 0.0
    ref_transforms = get_spec_transform_paths(spec_ref)
    if len(ref_transforms) > 0:
        w_transform = 1.0
        x_transform = w_transform * spec_f1_correctness_transform(spec_ref, spec_hyp).f1
    else:
        hyp_transforms = get_spec_transform_paths(spec_hyp)
        if len(hyp_transforms) > 0:
            w_transform = 1.0
            x_transform = 0.0
            w_encoding_penalty += 0.25
        else:
            w_transform = w_drawable
            x_transform = w_transform

    utterance_marks = get_marks_in_utterance(utterance)
    if not utterance_marks and _prompt_mentions_mark_fallback(utterance):
        utterance_marks = ["mentioned"]
    w_mark = 1.0 if len(utterance_marks) > 0 else 0.5
    x_mark = w_mark * spec_f1_correctness_mark(spec_ref, spec_hyp).f1

    w_encoding = max(0.0, 1.0 - w_encoding_penalty) * 3.0
    x_encoding = w_encoding * spec_f1_correctness_encoding(spec_ref, spec_hyp).f1

    score = x_drawable + x_valid_schema + x_not_empty + x_encoding + x_mark + x_transform
    weight_sum = w_drawable + w_valid_schema + w_not_empty + w_encoding + w_mark + w_transform
    return score / weight_sum if weight_sum > 0 else 0.0


def compute_spec_metrics(spec_ref: SpecType, spec_hyp: SpecType) -> dict[str, float]:
    mark = spec_f1_correctness_mark(spec_ref, spec_hyp)
    encoding = spec_f1_correctness_encoding(spec_ref, spec_hyp)
    transform = spec_f1_correctness_transform(spec_ref, spec_hyp)
    full = spec_f1_correctness_full(spec_ref, spec_hyp)
    keys = spec_f1_correctness_key(spec_ref, spec_hyp)
    kvs = spec_f1_correctness_key_values(spec_ref, spec_hyp)
    return {
        **{f"mark_{k}": v for k, v in mark._asdict().items()},
        **{f"encoding_{k}": v for k, v in encoding._asdict().items()},
        **{f"transform_{k}": v for k, v in transform._asdict().items()},
        **{f"full_{k}": v for k, v in full._asdict().items()},
        **{f"keys_{k}": v for k, v in keys._asdict().items()},
        **{f"kvs_{k}": v for k, v in kvs._asdict().items()},
        "keys_jaccard": spec_jaccard_keys(spec_ref, spec_hyp),
    }


def compute_vegachat_spec_score(
    spec_ref: SpecType,
    spec_hyp: SpecType,
    *,
    utterance: str,
    hyp_is_drawable: bool,
    hyp_is_empty_chart: bool,
    hyp_is_valid_schema: bool,
) -> VegaChatSpecScoreResult:
    mark = spec_f1_correctness_mark(spec_ref, spec_hyp)
    encoding = spec_f1_correctness_encoding(spec_ref, spec_hyp)
    transform = spec_f1_correctness_transform(spec_ref, spec_hyp)
    full = spec_f1_correctness_full(spec_ref, spec_hyp)
    keys = spec_f1_correctness_key(spec_ref, spec_hyp)
    kvs = spec_f1_correctness_key_values(spec_ref, spec_hyp)
    keys_jaccard = spec_jaccard_keys(spec_ref, spec_hyp)
    score = spec_score_impl(
        spec_ref,
        spec_hyp,
        utterance=utterance,
        hyp_is_drawable=hyp_is_drawable,
        hyp_is_empty_chart=hyp_is_empty_chart,
        hyp_is_valid_schema=hyp_is_valid_schema,
    )
    return VegaChatSpecScoreResult(
        spec_score=score,
        mark=mark,
        encoding=encoding,
        transform=transform,
        full=full,
        keys=keys,
        kvs=kvs,
        keys_jaccard=keys_jaccard,
        is_drawable=hyp_is_drawable,
        is_empty_chart=hyp_is_empty_chart,
        is_valid_schema=hyp_is_valid_schema,
        details=[
            "formula=vegachat_spec_score_impl",
            f"hyp_is_drawable={hyp_is_drawable}",
            f"hyp_is_empty_chart={hyp_is_empty_chart}",
            f"hyp_is_valid_schema={hyp_is_valid_schema}",
            f"mark_f1={mark.f1:.6f}",
            f"encoding_f2={encoding.f1:.6f}",
            f"transform_f1={transform.f1:.6f}",
            f"full_f1={full.f1:.6f}",
            f"keys_jaccard={_safe_float(keys_jaccard):.6f}",
        ],
    )


def _safe_float(value: float) -> float:
    return value if not math.isnan(value) else 0.0


def mean_metric(values: list[float]) -> float:
    cleaned = [value for value in values if not math.isnan(value)]
    if not cleaned:
        return float("nan")
    return sum(cleaned) / len(cleaned)


def compute_metric_means(items: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item.keys()})
    out: dict[str, float] = {}
    for key in keys:
        out[key] = mean_metric([float(item[key]) for item in items if key in item and item[key] is not None])
    return out
