from __future__ import annotations

from src.domain.models import DataColumnProfile, DataProfile
from src.orchestrator.analysis_planner import AnalysisPlanner


def _profile() -> DataProfile:
    return DataProfile(
        row_count=120,
        col_count=5,
        columns=[
            DataColumnProfile(name="sample_id", dtype="categorical", role="identifier", unique_count=120, is_identifier=True),
            DataColumnProfile(name="condition", dtype="categorical", role="dimension", unique_count=3),
            DataColumnProfile(name="score", dtype="numeric", role="measure", unique_count=80, outlier_count=2),
            DataColumnProfile(name="age", dtype="numeric", role="measure", unique_count=55),
            DataColumnProfile(name="date", dtype="datetime", role="temporal", unique_count=30),
        ],
    )


def test_general_query_creates_at_most_three_executable_subtasks() -> None:
    plan = AnalysisPlanner(max_charts=3).plan(
        user_query="Проанализируй данные и покажи основные закономерности",
        data_path="data.csv",
        data_profile=_profile(),
    )

    assert 1 <= len(plan.subtasks) <= 3
    available = {column.name for column in _profile().columns}
    assert all(set(item.required_fields).issubset(available) for item in plan.subtasks)
    assert len({item.task_type for item in plan.subtasks}) == len(plan.subtasks)
    assert any(item.task_type in {"group_comparison", "temporal_trend", "distribution", "correlation"} for item in plan.subtasks)


def test_specific_group_query_prioritizes_group_comparison() -> None:
    plan = AnalysisPlanner(max_charts=3).plan(
        user_query="Сравни score между condition",
        data_path="data.csv",
        data_profile=_profile(),
    )

    assert plan.subtasks[0].task_type == "group_comparison"
    assert plan.subtasks[0].required_fields == ["condition", "score"]
    assert plan.subtasks[0].constraints["output_target"] == "scientific_figure"


def test_planner_records_skipped_candidates_when_fields_are_missing() -> None:
    profile = DataProfile(
        row_count=10,
        col_count=1,
        columns=[DataColumnProfile(name="label", dtype="categorical", role="dimension", unique_count=3)],
    )

    plan = AnalysisPlanner(max_charts=3).plan(
        user_query="Покажи корреляцию и выбросы",
        data_path="labels.csv",
        data_profile=profile,
    )

    skipped = {item.task_type for item in plan.skipped_candidates}
    assert "correlation" in skipped
    assert "outlier_detection" in skipped
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].task_type == "overview"
