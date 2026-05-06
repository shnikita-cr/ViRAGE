from __future__ import annotations

from src.visrag_core.field_grounding import map_fields
from src.visrag_core.models import VisRAGColumnProfile, VisRAGDataProfile, VisRAGRequest


def _request(policy: str) -> VisRAGRequest:
    return VisRAGRequest(
        query="compare sales across categories",
        data_profile=VisRAGDataProfile(
            columns=[
                VisRAGColumnProfile(name="Category", semantic_type="nominal", role="dimension", raw_dtype="object"),
                VisRAGColumnProfile(name="Sales", semantic_type="quantitative", role="measure", raw_dtype="float64"),
                VisRAGColumnProfile(name="Month", semantic_type="temporal", role="date", raw_dtype="datetime64[ns]"),
            ]
        ),
        selected_fields=["Category", "Sales"],
        selected_fields_policy=policy,
    )


def test_strict_selected_fields_rejects_roles_not_available_inside_selected_fields() -> None:
    mapping, missing = map_fields(
        {"x": "temporal", "y": "quantitative"},
        _request("strict"),
    )

    assert mapping == {"y": "Sales"}
    assert missing == ["x"]


def test_prefer_selected_fields_can_fallback_to_compatible_non_selected_field() -> None:
    mapping, missing = map_fields(
        {"x": "temporal", "y": "quantitative"},
        _request("prefer"),
    )

    assert mapping == {"x": "Month", "y": "Sales"}
    assert missing == []


def test_auto_selected_fields_keeps_backward_compatible_prefer_behavior() -> None:
    mapping, missing = map_fields(
        {"x": "temporal", "y": "quantitative"},
        _request("auto"),
    )

    assert mapping == {"x": "Month", "y": "Sales"}
    assert missing == []
