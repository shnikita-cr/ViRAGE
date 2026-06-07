from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from src.benchmark.core.models import BenchmarkCase

SAMPLING_RANDOM = "random"
SAMPLING_STRATIFIED_CHART_TYPE = "stratified_chart_type"
SAMPLING_STRATEGIES = (SAMPLING_RANDOM, SAMPLING_STRATIFIED_CHART_TYPE)

SAMPLING_ALLOCATION_ROUND_ROBIN = "round_robin"
SAMPLING_ALLOCATION_BALANCED = "balanced"
SAMPLING_ALLOCATIONS = (SAMPLING_ALLOCATION_ROUND_ROBIN, SAMPLING_ALLOCATION_BALANCED)

_CHART_TYPE_ALIASES = {
    "singleattrbar": "bar",
    "bar": "bar",
    "barchart": "bar",
    "bar_chart": "bar",
    "scatterfaceted": "scatter",
    "scatter": "scatter",
    "scatterplot": "scatter",
    "scatter_plot": "scatter",
    "point": "scatter",
    "circle": "scatter",
    "line": "line",
    "linechart": "line",
    "line_chart": "line",
    "histogram": "histogram",
    "hist": "histogram",
    "boxplot": "boxplot",
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
    sampling_allocation: str = SAMPLING_ALLOCATION_ROUND_ROBIN
    cases_per_chart_type: int | None = None
    case_ids_hash: str = ""
    sampling_warning: str | None = None

    def as_report_payload(self) -> dict[str, Any]:
        return {
            "sampling_strategy": self.sampling_strategy,
            "sampling_allocation": self.sampling_allocation,
            "cases_per_chart_type": self.cases_per_chart_type,
            "seed": self.seed,
            "limit": self.limit,
            "selected_case_ids": list(self.selected_case_ids),
            "selected_chart_types": list(self.selected_chart_types),
            "chart_type_distribution": dict(self.chart_type_distribution),
            "case_ids_hash": self.case_ids_hash,
            "sampling_warning": self.sampling_warning,
        }


def select_benchmark_cases(
    cases: list[BenchmarkCase],
    *,
    limit: int | None,
    shuffle: bool,
    seed: int,
    sampling_strategy: str = SAMPLING_RANDOM,
    max_per_chart_type: int | None = None,
    sampling_allocation: str = SAMPLING_ALLOCATION_ROUND_ROBIN,
    cases_per_chart_type: int | None = None,
) -> tuple[list[BenchmarkCase], BenchmarkSamplingSummary]:
    unique_cases = _dedupe_case_ids(cases)
    strategy = _normalize_sampling_strategy(sampling_strategy)
    allocation = _normalize_sampling_allocation(sampling_allocation)
    selected = _select_cases(
        unique_cases,
        limit=limit,
        shuffle=shuffle,
        seed=seed,
        sampling_strategy=strategy,
        max_per_chart_type=max_per_chart_type,
        sampling_allocation=allocation,
        cases_per_chart_type=cases_per_chart_type,
    )
    warning = _sampling_warning(selected=selected, requested_limit=limit, requested_per_type=cases_per_chart_type)
    return selected, _summary(
        selected,
        strategy=strategy,
        allocation=allocation,
        seed=seed,
        limit=limit,
        cases_per_chart_type=cases_per_chart_type,
        warning=warning,
    )


def chart_type_from_case(case: BenchmarkCase) -> str:
    case_id_type = _chart_type_from_case_id(case.case_id)
    if case_id_type != "unknown":
        return case_id_type
    explicit = _clean_chart_type(case.metadata.get("chart_type"))
    if explicit != "unknown":
        return explicit
    return _chart_type_from_spec(case.reference_spec)


def _select_cases(
    cases: list[BenchmarkCase],
    *,
    limit: int | None,
    shuffle: bool,
    seed: int,
    sampling_strategy: str,
    max_per_chart_type: int | None,
    sampling_allocation: str,
    cases_per_chart_type: int | None,
) -> list[BenchmarkCase]:
    if sampling_strategy == SAMPLING_STRATIFIED_CHART_TYPE:
        return _select_stratified_by_chart_type(
            cases,
            limit=limit,
            seed=seed,
            max_per_chart_type=max_per_chart_type,
            sampling_allocation=sampling_allocation,
            cases_per_chart_type=cases_per_chart_type,
        )
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
    sampling_allocation: str,
    cases_per_chart_type: int | None,
) -> list[BenchmarkCase]:
    rng = random.Random(seed)
    groups = _group_by_chart_type(cases)
    for grouped_cases in groups.values():
        rng.shuffle(grouped_cases)
    if sampling_allocation == SAMPLING_ALLOCATION_BALANCED:
        selected = _select_balanced(groups, cases_per_chart_type=cases_per_chart_type, max_per_chart_type=max_per_chart_type)
    else:
        target = len(cases) if limit is None else max(0, limit)
        selected = _select_round_robin(groups, target=target, max_per_chart_type=max_per_chart_type)
    if limit is not None and cases_per_chart_type is None:
        return selected[: max(0, limit)]
    return selected


def _select_balanced(
    groups: dict[str, list[BenchmarkCase]],
    *,
    cases_per_chart_type: int | None,
    max_per_chart_type: int | None,
) -> list[BenchmarkCase]:
    per_type = cases_per_chart_type if cases_per_chart_type is not None else max_per_chart_type
    if per_type is None:
        smallest_group = min((len(group) for group in groups.values()), default=0)
        per_type = smallest_group
    target_per_type = max(0, int(per_type))
    selected: list[BenchmarkCase] = []
    for chart_type in sorted(groups):
        selected.extend(groups[chart_type][:target_per_type])
    return selected


def _select_round_robin(
    groups: dict[str, list[BenchmarkCase]],
    *,
    target: int,
    max_per_chart_type: int | None,
) -> list[BenchmarkCase]:
    chart_types = sorted(groups)
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
    allocation: str,
    seed: int,
    limit: int | None,
    cases_per_chart_type: int | None,
    warning: str | None,
) -> BenchmarkSamplingSummary:
    chart_types = [chart_type_from_case(case) for case in selected]
    case_ids = [case.case_id for case in selected]
    return BenchmarkSamplingSummary(
        sampling_strategy=strategy,
        sampling_allocation=allocation,
        cases_per_chart_type=cases_per_chart_type,
        seed=seed,
        limit=limit,
        selected_case_ids=case_ids,
        selected_chart_types=chart_types,
        chart_type_distribution=dict(sorted(Counter(chart_types).items())),
        case_ids_hash=_hash_case_ids(case_ids),
        sampling_warning=warning,
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


def _normalize_sampling_allocation(value: str) -> str:
    allocation = str(value or SAMPLING_ALLOCATION_ROUND_ROBIN).strip().lower().replace("-", "_")
    if allocation not in SAMPLING_ALLOCATIONS:
        raise ValueError(f"Unsupported benchmark sampling allocation: {value}")
    return allocation


def _chart_type_from_spec(spec: dict[str, Any]) -> str:
    for node in _walk_spec(spec):
        if not isinstance(node, dict):
            continue
        mark_type = _mark_type(node.get("mark"))
        if mark_type == "bar" and _has_binned_encoding(node):
            return "histogram"
        if mark_type != "unknown":
            return _clean_chart_type(mark_type)
    return "unknown"


def _mark_type(mark: Any) -> str:
    if isinstance(mark, str):
        return _clean_chart_type(mark)
    if isinstance(mark, dict) and isinstance(mark.get("type"), str):
        return _clean_chart_type(mark["type"])
    return "unknown"


def _has_binned_encoding(node: dict[str, Any]) -> bool:
    encoding = node.get("encoding")
    if not isinstance(encoding, dict):
        return False
    for channel in encoding.values():
        if isinstance(channel, dict) and "bin" in channel:
            return True
    return False


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


def _hash_case_ids(case_ids: list[str]) -> str:
    payload = "\n".join(case_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _sampling_warning(*, selected: list[BenchmarkCase], requested_limit: int | None, requested_per_type: int | None) -> str | None:
    if requested_per_type is not None:
        return None
    if requested_limit is not None and len(selected) < requested_limit:
        return f"selected {len(selected)} cases, requested {requested_limit}"
    return None
