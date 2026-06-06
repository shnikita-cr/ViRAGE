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




class FakePlannerTextLLM:
    model = "fake-planner"

    def __init__(self, content: str) -> None:
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, messages: Any) -> FakeLLMResult:
        self.prompts.append(str(messages))
        return FakeLLMResult(self.content)


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

def test_llm_planner_accepts_markdown_fenced_json_from_local_model() -> None:
    content = "```json\n" + json.dumps(_valid_payload(), ensure_ascii=False) + "\n```"

    plan = AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerTextLLM(content), max_attempts=1).plan(
        user_query="Проанализируй данные",
        data_path="data.csv",
        data_profile=_profile(),
    )

    assert len(plan.subtasks) == 2
    assert plan.subtasks[0].required_fields == ["condition", "score"]


def test_llm_planner_accepts_json_embedded_in_text_from_local_model() -> None:
    content = "Here is the plan:\n" + json.dumps(_valid_payload(), ensure_ascii=False)

    plan = AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerTextLLM(content), max_attempts=1).plan(
        user_query="Проанализируй данные",
        data_path="data.csv",
        data_profile=_profile(),
    )

    assert plan.subtasks[1].task_type == "temporal_trend"
    assert plan.subtasks[1].required_fields == ["date", "score"]


def test_llm_planner_rejects_text_without_valid_json() -> None:
    with pytest.raises(RuntimeError, match="no valid JSON object could be extracted"):
        AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerTextLLM("I cannot build this plan."), max_attempts=1).plan(
            user_query="Проанализируй данные",
            data_path="data.csv",
            data_profile=_profile(),
        )



def test_llm_planner_fills_runtime_fields_and_normalizes_rationale_string() -> None:
    payload = json.loads(json.dumps(_valid_payload(), ensure_ascii=False))
    payload.pop("user_query")
    payload.pop("data_path")
    payload["rationale"] = "The plan uses existing fields."

    plan = AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerLLM(payload), max_attempts=1).plan(
        user_query="Сравни score между condition",
        data_path="runtime.csv",
        data_profile=_profile(),
    )

    assert plan.user_query == "Сравни score между condition"
    assert plan.data_path == "runtime.csv"
    assert plan.rationale == ["The plan uses existing fields."]


def test_llm_planner_normalizes_invalid_severity_scale_from_local_model() -> None:
    payload = json.loads(json.dumps(_valid_payload(), ensure_ascii=False))
    payload["subtasks"] = [
        {
            "id": "analysis_001",
            "task_type": "ranking",
            "query": "Rank the worst outliers by severity.",
            "purpose": "Find the most problematic quality issues.",
            "required_fields": ["condition", "score"],
            "optional_fields": [],
            "priority": 1,
            "constraints": {"output_target": "scientific_figure"},
            "metric_semantics": {"score": "higher_is_worse"},
            "ranking_strategy": "top_n_highest_severity",
            "scale_strategy": "raw_values",
            "visual_constraints": [],
            "rationale": "The user asked for problematic outliers.",
        }
    ]

    plan = AnalysisPlanner(max_charts=3, reasoning_llm=FakePlannerLLM(payload), max_attempts=1).plan(
        user_query="Покажи худшие выбросы",
        data_path="data.csv",
        data_profile=_profile(),
    )

    assert plan.subtasks[0].scale_strategy == "normalized_severity"
    assert "use_normalized_severity_for_problem_ranking" in plan.subtasks[0].visual_constraints
