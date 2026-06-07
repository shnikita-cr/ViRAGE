from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from src.benchmark.core.models import BenchmarkCase

SAMPLING_RANDOM = "random"
SAMPLING_STRATIFIED_CHART_TYPE = "stratified_chart_type"
SAMPLING_STRATEGIES = (SAMPLING_RANDOM, SAMPLING_STRATIFIED_CHART_TYPE)

_CHART_TYPE_ALIASES = {
    "singleattrbar": "bar",
    "bar": "bar",
    "barchart": "bar",
    "bar_chart": "bar",
    "scatterfaceted": "scatter",
    "scatterplot": "scatter",
    "scatter_plot": "scatter",
    "point": "scatter",
    "circle": "scatter",
    "linechart": "line",
    "line_chart": "line",
    "hist": "histogram",
    "box": "boxplot",
    "box_plot": "boxplot",
}


@dataclass(frozen=True, slots=True)
class BenchmarkSamplingSummary:
    sampling_strategy: str
    seed: int
    limit: int | None
    selected_case_ids: list[str]
    selected_chart_types: list[str]
    chart_type_distribution: dict[str, int]

    def as_report_payload(self) -> dict[str, Any]:
        return {
            "sampling_strategy": self.sampling_strategy,
            "seed": self.seed,
            "limit": self.limit,
            "selected_case_ids": list(self.selected_case_ids),
            "selected_chart_types": list(self.selected_chart_types),
            "chart_type_distribution": dict(self.chart_type_distribution),
        }


def select_benchmark_cases(
        cases: list[BenchmarkCase],
        *,
        limit: int | None,
        shuffle: bool,
        seed: int,
        sampling_strategy: str = SAMPLING_RANDOM,
        max_per_chart_type: int | None = None,
) -> tuple[list[BenchmarkCase], BenchmarkSamplingSummary]:
    unique_cases = _dedupe_case_ids(cases)
    strategy = _normalize_sampling_strategy(sampling_strategy)
    selected = _select_cases(
        unique_cases,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
        sampling_strategy=strategy,
        max_per_chart_type=max_per_chart_type,
    )
    return selected, _summary(selected, strategy=strategy, seed=seed, limit=limit)


def chart_type_from_case(case: BenchmarkCase) -> str:
    explicit = _clean_chart_type(case.metadata.get("chart_type"))
    if explicit != "unknown":
        return explicit
    spec_type = _chart_type_from_spec(case.reference_spec)
    if spec_type != "unknown":
        return spec_type
    return _chart_type_from_case_id(case.case_id)


def _select_cases(
        cases: list[BenchmarkCase],
        *,
        limit: int | None,
        shuffle: bool,
        seed: int,
        sampling_strategy: str,
        max_per_chart_type: int | None,
) -> list[BenchmarkCase]:
    if sampling_strategy == SAMPLING_STRATIFIED_CHART_TYPE:
        return _select_stratified_by_chart_type(cases, limit=limit, seed=seed, max_per_chart_type=max_per_chart_type)
    return _select_random(cases, limit=limit, shuffle=shuffle, seed=seed)


def _select_random(cases: list[BenchmarkCase], *, limit: int | None, shuffle: bool, seed: int) -> list[BenchmarkCase]:
    selected = list(cases)
    if shuffle:
        random.Random(seed).shuffle(selected)
    if limit is None:
        return selected
    return selected[: max(0, limit)]


def _select_stratified_by_chart_type(
        cases: list[BenchmarkCase],
        *,
        limit: int | None,
        seed: int,
        max_per_chart_type: int | None,
) -> list[BenchmarkCase]:
    target = len(cases) if limit is None else max(0, limit)
    rng = random.Random(seed)
    groups = _group_by_chart_type(cases)
    for grouped_cases in groups.values():
        rng.shuffle(grouped_cases)
    chart_types = sorted(groups)
    rng.shuffle(chart_types)
    selected: list[BenchmarkCase] = []
    per_type_counts: Counter[str] = Counter()
    while len(selected) < target:
        added_in_round = False
        for chart_type in chart_types:
            if len(selected) >= target:
                break
            if max_per_chart_type is not None and per_type_counts[chart_type] >= max_per_chart_type:
                continue
            grouped_cases = groups[chart_type]
            index = per_type_counts[chart_type]
            if index >= len(grouped_cases):
                continue
            selected.append(grouped_cases[index])
            per_type_counts[chart_type] += 1
            added_in_round = True
        if not added_in_round:
            break
    return selected


def _group_by_chart_type(cases: list[BenchmarkCase]) -> dict[str, list[BenchmarkCase]]:
    groups: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        groups[chart_type_from_case(case)].append(case)
    return dict(groups)


def _summary(
        selected: list[BenchmarkCase],
        *,
        strategy: str,
        seed: int,
        limit: int | None,
) -> BenchmarkSamplingSummary:
    chart_types = [chart_type_from_case(case) for case in selected]
    return BenchmarkSamplingSummary(
        sampling_strategy=strategy,
        seed=seed,
        limit=limit,
        selected_case_ids=[case.case_id for case in selected],
        selected_chart_types=chart_types,
        chart_type_distribution=dict(sorted(Counter(chart_types).items())),
    )


def _dedupe_case_ids(cases: list[BenchmarkCase]) -> list[BenchmarkCase]:
    selected: list[BenchmarkCase] = []
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        selected.append(case)
    return selected


def _normalize_sampling_strategy(value: str) -> str:
    strategy = str(value or SAMPLING_RANDOM).strip().lower().replace("-", "_")
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"Unsupported benchmark sampling strategy: {value}")
    return strategy


def _chart_type_from_spec(spec: dict[str, Any]) -> str:
    for node in _walk_spec(spec):
        if not isinstance(node, dict):
            continue
        mark = node.get("mark")
        if isinstance(mark, str):
            return _clean_chart_type(mark)
        if isinstance(mark, dict):
            mark_type = mark.get("type")
            if isinstance(mark_type, str):
                return _clean_chart_type(mark_type)
    return "unknown"


def _chart_type_from_case_id(case_id: str) -> str:
    normalized = str(case_id).strip().lower().replace("-", "_")
    for token, chart_type in sorted(_CHART_TYPE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if token in normalized:
            return chart_type
    return "unknown"


def _clean_chart_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _CHART_TYPE_ALIASES.get(text, text or "unknown")


def _walk_spec(node: Any):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_spec(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_spec(item)
