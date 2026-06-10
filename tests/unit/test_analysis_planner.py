from __future__ import annotations

import json
import re
from typing import Any

import pytest

from src.domain.models import DataColumnProfile, DataProfile
from src.orchestrator.analysis_planner import AnalysisPlanner


class FakeLLMResult:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1}


class FakePlannerLLM:
    model = "fake-planner"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def invoke(self, messages: Any) -> FakeLLMResult:
        self.prompts.append(str(messages))
        return FakeLLMResult(json.dumps(self.payload, ensure_ascii=False))


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


def _valid_payload() -> dict[str, Any]:
    return {
        "user_query": "Проанализируй данные",
        "data_path": "data.csv",
        "input_type": "table",
        "max_charts": 3,
        "subtasks": [
            {
                "id": "group_001",
                "task_type": "group_comparison",
                "query": "Compare score across condition.",
                "purpose": "Compare the main measure between groups.",
                "required_fields": ["condition", "score"],
                "optional_fields": [],
                "priority": 1,
                "constraints": {"output_target": "scientific_figure"},
                "rationale": "Both fields exist and match the request.",
            },
            {
                "id": "trend_001",
                "task_type": "temporal_trend",
                "query": "Show score over date.",
                "purpose": "Check temporal dynamics of the score.",
                "required_fields": ["date", "score"],
                "optional_fields": [],
                "priority": 2,
                "constraints": {"output_target": "scientific_figure"},
                "rationale": "A temporal field and measure are available.",
            },
        ],
        "skipped_candidates": [
            {"task_type": "correlation", "reason": "The request did not prioritize a relationship view.", "required_fields": []}
        ],
        "rationale": ["The plan stays within max_charts and uses existing fields."],
    }


def test_llm_planner_accepts_strict_valid_plan() -> None:
    plan = AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerLLM(_valid_payload())).plan(
        user_query="Проанализируй данные и покажи основные закономерности",
        data_path="data.csv",
        data_profile=_profile(),
    )

    assert len(plan.subtasks) == 2
    available = {column.name for column in _profile().columns}
    assert all(set(item.required_fields).issubset(available) for item in plan.subtasks)
    assert plan.subtasks[0].task_type == "group_comparison"
    assert plan.subtasks[0].required_fields == ["condition", "score"]


def test_llm_planner_requires_reasoning_llm() -> None:
    with pytest.raises(RuntimeError, match="reasoning LLM"):
        AnalysisPlanner(max_charts=3).plan(
            user_query="Проанализируй данные",
            data_path="data.csv",
            data_profile=_profile(),
        )


def test_llm_planner_rejects_invented_fields() -> None:
    payload = _valid_payload()
    payload["subtasks"][0]["required_fields"] = ["condition", "invented_score"]

    with pytest.raises(RuntimeError, match="absent from DataProfile"):
        AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerLLM(payload)).plan(
            user_query="Проанализируй данные",
            data_path="data.csv",
            data_profile=_profile(),
        )


def test_llm_planner_rejects_empty_required_fields() -> None:
    payload = _valid_payload()
    payload["subtasks"][0]["required_fields"] = []

    with pytest.raises(RuntimeError, match="Failed to parse AnalysisPlan"):
        AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerLLM(payload), max_attempts=1).plan(
            user_query="Проанализируй данные",
            data_path="data.csv",
            data_profile=_profile(),
        )




def test_llm_planner_rejects_more_subtasks_than_runtime_max_charts() -> None:
    payload = json.loads(json.dumps(_valid_payload()))
    payload["subtasks"].append(
        {
            "id": "corr_001",
            "task_type": "correlation",
            "query": "Analyze the relationship between age and score.",
            "purpose": "Check whether age relates to score.",
            "required_fields": ["age", "score"],
            "optional_fields": [],
            "priority": 3,
            "constraints": {"output_target": "scientific_figure"},
            "rationale": "Both numeric fields exist in the DataProfile.",
        }
    )

    with pytest.raises(RuntimeError, match="Failed to parse AnalysisPlan"):
        AnalysisPlanner(max_charts=2, reasoning_llm=FakePlannerLLM(payload), max_attempts=1).plan(
            user_query="Проанализируй данные",
            data_path="data.csv",
            data_profile=_profile(),
        )


def test_llm_planner_prompt_contains_no_rules_mode() -> None:
    llm = FakePlannerLLM(_valid_payload())
    AnalysisPlanner(max_charts=3, reasoning_llm=llm).plan(
        user_query="Сравни score между condition",
        data_path="data.csv",
        data_profile=_profile(),
    )

    assert llm.prompts
    assert "rules planner has been removed" not in llm.prompts[0]
    assert re.search(r"available_fields", llm.prompts[0])

class FencedPlannerLLM(FakePlannerLLM):
    def invoke(self, messages: Any) -> FakeLLMResult:
        self.prompts.append(str(messages))
        return FakeLLMResult("```json\n" + json.dumps(self.payload, ensure_ascii=False) + "\n```")


def test_llm_planner_accepts_json_inside_markdown_fence() -> None:
    plan = AnalysisPlanner(max_charts=3, reasoning_llm=FencedPlannerLLM(_valid_payload())).plan(
        user_query="Проанализируй данные",
        data_path="data.csv",
        data_profile=_profile(),
    )

    assert len(plan.subtasks) == 2
    assert plan.data_path == "data.csv"


def test_llm_planner_fills_runtime_fields_when_model_omits_them() -> None:
    payload = _valid_payload()
    payload.pop("user_query")
    payload.pop("data_path")
    payload["rationale"] = "The selected fields exist."

    plan = AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerLLM(payload)).plan(
        user_query="Проанализируй данные",
        data_path="data.csv",
        data_profile=_profile(),
    )

    assert plan.user_query == "Проанализируй данные"
    assert plan.data_path == "data.csv"
    assert plan.rationale == ["The selected fields exist."]


def _image_profile() -> DataProfile:
    return DataProfile(
        row_count=4,
        col_count=4,
        columns=[
            DataColumnProfile(name="file_name", dtype="categorical", role="identifier", unique_count=4, is_identifier=True),
            DataColumnProfile(name="group", dtype="categorical", role="dimension", unique_count=2),
            DataColumnProfile(name="brisque_score", dtype="numeric", role="measure", unique_count=4),
            DataColumnProfile(name="laplacian_variance", dtype="numeric", role="measure", unique_count=4),
        ],
    )


def test_llm_planner_accepts_image_quality_comparison_without_problem_ranking() -> None:
    payload = {
        "user_query": "Сравни качество изображений между группами",
        "data_path": "image_quality_metrics.csv",
        "input_type": "image_folder",
        "max_charts": 3,
        "subtasks": [
            {
                "id": "method_iqa_comparison",
                "task_type": "image_quality_analysis",
                "query": "Compare image quality metrics between groups.",
                "purpose": "Compare image quality metrics between groups.",
                "required_fields": ["group", "brisque_score", "laplacian_variance"],
                "optional_fields": [],
                "priority": 1,
                "constraints": {"output_target": "scientific_figure"},
                "metric_semantics": {"brisque_score": "lower_is_better", "laplacian_variance": "higher_is_better"},
                "ranking_strategy": None,
                "scale_strategy": "independent_panels",
                "visual_constraints": ["use_independent_panels_for_multimetric"],
                "rationale": "The fields are present and support group-level IQA comparison.",
            }
        ],
        "skipped_candidates": [
            {"task_type": "outlier_detection", "reason": "The request does not ask for problematic files.", "required_fields": []}
        ],
        "rationale": ["The plan compares available image metrics by group."],
    }

    plan = AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerLLM(payload), max_attempts=1).plan(
        user_query="Сравни качество изображений между группами",
        data_path="image_quality_metrics.csv",
        data_profile=_image_profile(),
        input_type="image_folder",
    )

    assert plan.subtasks[0].id == "method_iqa_comparison"
    assert plan.subtasks[0].ranking_strategy is None


def test_llm_planner_still_rejects_problematic_items_without_problem_ranking() -> None:
    payload = {
        "user_query": "Найди проблемные изображения",
        "data_path": "image_quality_metrics.csv",
        "input_type": "image_folder",
        "max_charts": 3,
        "subtasks": [
            {
                "id": "problem_images",
                "task_type": "image_quality_analysis",
                "query": "Find problematic image files.",
                "purpose": "Find problematic items in the image collection.",
                "required_fields": ["file_name", "brisque_score"],
                "optional_fields": [],
                "priority": 1,
                "constraints": {"output_target": "scientific_figure"},
                "metric_semantics": {"brisque_score": "lower_is_better"},
                "ranking_strategy": None,
                "scale_strategy": "normalized_severity",
                "visual_constraints": ["use_overall_severity_for_problematic_items"],
                "rationale": "The fields are present and support problematic item ranking.",
            }
        ],
        "skipped_candidates": [],
        "rationale": ["The plan targets problematic images."],
    }

    with pytest.raises(RuntimeError, match="problematic items"):
        AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerLLM(payload), max_attempts=1).plan(
            user_query="Найди проблемные изображения",
            data_path="image_quality_metrics.csv",
            data_profile=_image_profile(),
            input_type="image_folder",
        )
