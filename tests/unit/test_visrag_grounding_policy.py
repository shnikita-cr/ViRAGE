from __future__ import annotations

from src.visrag_core.grounding_policy import (
    GroundingPolicyResolver,
    SelectedFieldsPolicy,
    resolve_grounding_policy,
)
from src.visrag_core.models import VisRAGColumnProfile, VisRAGDataProfile, VisRAGRequest


def _request(
    *,
    query: str,
    selected_fields: list[str],
    preferred_chart_types: list[str] | None = None,
    columns: list[VisRAGColumnProfile] | None = None,
) -> VisRAGRequest:
    return VisRAGRequest(
        query=query,
        data_profile=VisRAGDataProfile(
            columns=columns
            or [
                VisRAGColumnProfile(name="Category", semantic_type="object", role="dimension", raw_dtype="object"),
                VisRAGColumnProfile(name="Sales", semantic_type="float64", role="measure", raw_dtype="float64"),
                VisRAGColumnProfile(name="Month", semantic_type="datetime64[ns]", role="date", raw_dtype="datetime64[ns]"),
            ]
        ),
        preferred_chart_types=preferred_chart_types or [],
        selected_fields=selected_fields,
        top_k=5,
    )


def test_resolver_prefers_when_selected_fields_are_empty() -> None:
    decision = resolve_grounding_policy(
        _request(
            query="show me a useful chart",
            selected_fields=[],
        ),
        request_confidence=0.3,
    )

    assert decision.selected_fields_policy == SelectedFieldsPolicy.PREFER
    assert "no_selected_fields" in decision.reasons


def test_resolver_uses_strict_for_high_confidence_explicit_category_comparison() -> None:
    decision = resolve_grounding_policy(
        _request(
            query="compare sales across product categories with a bar chart",
            selected_fields=["Category", "Sales"],
            preferred_chart_types=["bar"],
        ),
        request_confidence=0.92,
    )

    assert decision.selected_fields_policy == SelectedFieldsPolicy.STRICT
    assert "selected_fields_present" in decision.reasons
    assert "request_confidence_high" in decision.reasons
    assert "explicit_query_signal" in decision.reasons


def test_resolver_uses_strict_for_high_confidence_trend_with_preferred_chart() -> None:
    decision = resolve_grounding_policy(
        _request(
            query="show sales over time",
            selected_fields=["Month", "Sales"],
            preferred_chart_types=["line"],
        ),
        request_confidence=0.81,
    )

    assert decision.selected_fields_policy == SelectedFieldsPolicy.STRICT
    assert "preferred_chart_types_present" in decision.reasons


def test_resolver_uses_soft_fail_for_medium_confidence_selected_fields() -> None:
    decision = resolve_grounding_policy(
        _request(
            query="show distribution",
            selected_fields=["Sales"],
            preferred_chart_types=["histogram"],
        ),
        request_confidence=0.61,
    )

    assert decision.selected_fields_policy == SelectedFieldsPolicy.SOFT_FAIL
    assert "request_confidence_medium" in decision.reasons


def test_resolver_uses_soft_fail_when_selected_field_is_missing_from_profile() -> None:
    decision = GroundingPolicyResolver().resolve(
        _request(
            query="compare revenue by category",
            selected_fields=["Category", "Revenue"],
            preferred_chart_types=["bar"],
        ),
        request_confidence=0.95,
    )

    assert decision.selected_fields_policy == SelectedFieldsPolicy.SOFT_FAIL
    assert "missing_selected_fields" in decision.reasons
    assert "missing:Revenue" in decision.reasons


def test_resolver_uses_soft_fail_for_identifier_only_selected_fields() -> None:
    decision = resolve_grounding_policy(
        _request(
            query="compare records by id",
            selected_fields=["order_id"],
            preferred_chart_types=["bar"],
            columns=[
                VisRAGColumnProfile(name="order_id", semantic_type="object", role="identifier", raw_dtype="object"),
                VisRAGColumnProfile(name="Sales", semantic_type="float64", role="measure", raw_dtype="float64"),
            ],
        ),
        request_confidence=0.9,
    )

    assert decision.selected_fields_policy == SelectedFieldsPolicy.SOFT_FAIL
    assert "selected_fields_identifier_like_only" in decision.reasons


def test_resolver_uses_soft_fail_when_ambiguity_notes_are_present() -> None:
    decision = resolve_grounding_policy(
        _request(
            query="show distribution by date and category",
            selected_fields=["Month", "Category"],
            preferred_chart_types=["bar", "line", "histogram"],
        ),
        request_confidence=0.88,
        ambiguity_notes=["ambiguous chart family"],
    )

    assert decision.selected_fields_policy == SelectedFieldsPolicy.SOFT_FAIL
    assert "ambiguity_notes_present" in decision.reasons


class _RequestWithOverride(VisRAGRequest):
    selected_fields_policy: str = "strict"


def test_explicit_policy_override_is_respected() -> None:
    request = _RequestWithOverride(
        query="show me a useful chart",
        data_profile=VisRAGDataProfile(
            columns=[
                VisRAGColumnProfile(name="Category", semantic_type="object", role="dimension", raw_dtype="object"),
                VisRAGColumnProfile(name="Sales", semantic_type="float64", role="measure", raw_dtype="float64"),
            ]
        ),
        selected_fields=[],
        preferred_chart_types=[],
    )

    decision = resolve_grounding_policy(request, request_confidence=0.1)

    assert decision.selected_fields_policy == SelectedFieldsPolicy.STRICT
    assert decision.reasons == ["explicit_policy_override"]
