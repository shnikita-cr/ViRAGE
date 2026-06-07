from __future__ import annotations

from src.benchmark.core.models import BenchmarkCase
from src.benchmark.core.sampling import SAMPLING_STRATIFIED_CHART_TYPE, chart_type_from_case, select_benchmark_cases


def _case(case_id: str, mark: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        query=f"make {mark}",
        data_path="data.csv",
        reference_spec={"mark": mark, "encoding": {}},
        metadata={},
    )


def test_random_sampling_never_returns_duplicate_case_ids() -> None:
    cases = [_case("a", "bar"), _case("a", "bar"), _case("b", "line")]

    selected, summary = select_benchmark_cases(cases, limit=3, shuffle=True, seed=7)

    assert [case.case_id for case in selected] == summary.selected_case_ids
    assert len(summary.selected_case_ids) == len(set(summary.selected_case_ids))


def test_stratified_sampling_prefers_distinct_chart_types() -> None:
    cases = [
        _case("bar_1", "bar"),
        _case("bar_2", "bar"),
        _case("line_1", "line"),
        _case("point_1", "point"),
    ]

    selected, summary = select_benchmark_cases(
        cases,
        limit=3,
        shuffle=True,
        seed=42,
        sampling_strategy=SAMPLING_STRATIFIED_CHART_TYPE,
        max_per_chart_type=1,
    )

    chart_types = [chart_type_from_case(case) for case in selected]
    assert len(selected) == 3
    assert len(chart_types) == len(set(chart_types))
    assert summary.chart_type_distribution == {chart_type: 1 for chart_type in sorted(chart_types)}


def test_stratified_sampling_is_seed_reproducible() -> None:
    cases = [_case(f"bar_{index}", "bar") for index in range(4)] + [_case(f"line_{index}", "line") for index in range(4)]

    first, first_summary = select_benchmark_cases(
        cases,
        limit=4,
        shuffle=True,
        seed=13,
        sampling_strategy=SAMPLING_STRATIFIED_CHART_TYPE,
    )
    second, second_summary = select_benchmark_cases(
        cases,
        limit=4,
        shuffle=True,
        seed=13,
        sampling_strategy=SAMPLING_STRATIFIED_CHART_TYPE,
    )

    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert first_summary.as_report_payload() == second_summary.as_report_payload()
