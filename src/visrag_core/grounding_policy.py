from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field

from .models import VisRAGColumnProfile, VisRAGDataProfile, VisRAGRequest
from .semantic import semantic_type_from_role_or_dtype


class SelectedFieldsPolicy(str, Enum):
    AUTO = "auto"
    PREFER = "prefer"
    STRICT = "strict"
    SOFT_FAIL = "soft_fail"


class GroundingPolicyDecision(BaseModel):
    selected_fields_policy: SelectedFieldsPolicy
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class GroundingPolicyResolver:
    """Deterministic resolver for selected field grounding behavior.

    The resolver does not run retrieval and does not call an LLM. It converts
    request-level signals into a concrete policy that field grounding can use
    later: prefer, strict, or soft_fail.
    """

    HIGH_CONFIDENCE_THRESHOLD = 0.75
    LOW_CONFIDENCE_THRESHOLD = 0.45
    MANY_SELECTED_FIELDS_THRESHOLD = 4

    def resolve(
        self,
        request: VisRAGRequest,
        *,
        request_confidence: float | None = None,
        ambiguity_notes: Iterable[str] | None = None,
    ) -> GroundingPolicyDecision:
        explicit_policy = _parse_policy(getattr(request, "selected_fields_policy", None))

        if explicit_policy and explicit_policy is not SelectedFieldsPolicy.AUTO:
            return GroundingPolicyDecision(
                selected_fields_policy=explicit_policy,
                confidence=1.0,
                reasons=["explicit_policy_override"],
            )

        selected_fields = _dedupe([field for field in request.selected_fields if field])
        column_by_name = {column.name: column for column in request.data_profile.columns}
        confidence = _clamp(request_confidence if request_confidence is not None else 0.5)
        ambiguity_values = [str(item).strip() for item in (ambiguity_notes or []) if str(item).strip()]
        reasons: list[str] = []

        if not selected_fields:
            return GroundingPolicyDecision(
                selected_fields_policy=SelectedFieldsPolicy.PREFER,
                confidence=_decision_confidence(confidence, 0.75),
                reasons=["no_selected_fields"],
            )

        reasons.append("selected_fields_present")

        missing_fields = [field for field in selected_fields if field not in column_by_name]
        if missing_fields:
            return GroundingPolicyDecision(
                selected_fields_policy=SelectedFieldsPolicy.SOFT_FAIL,
                confidence=_decision_confidence(confidence, 0.65),
                reasons=[*reasons, "missing_selected_fields", *_prefix_values("missing", missing_fields)],
            )

        if _only_identifier_like_fields(selected_fields, column_by_name):
            return GroundingPolicyDecision(
                selected_fields_policy=SelectedFieldsPolicy.SOFT_FAIL,
                confidence=_decision_confidence(confidence, 0.65),
                reasons=[*reasons, "selected_fields_identifier_like_only"],
            )

        if len(selected_fields) > self.MANY_SELECTED_FIELDS_THRESHOLD:
            return GroundingPolicyDecision(
                selected_fields_policy=SelectedFieldsPolicy.SOFT_FAIL,
                confidence=_decision_confidence(confidence, 0.6),
                reasons=[*reasons, "many_selected_fields"],
            )

        if ambiguity_values:
            return GroundingPolicyDecision(
                selected_fields_policy=SelectedFieldsPolicy.SOFT_FAIL,
                confidence=_decision_confidence(confidence, 0.6),
                reasons=[*reasons, "ambiguity_notes_present"],
            )

        if confidence < self.LOW_CONFIDENCE_THRESHOLD:
            return GroundingPolicyDecision(
                selected_fields_policy=SelectedFieldsPolicy.SOFT_FAIL,
                confidence=_decision_confidence(confidence, 0.55),
                reasons=[*reasons, "request_confidence_low"],
            )

        if confidence < self.HIGH_CONFIDENCE_THRESHOLD:
            return GroundingPolicyDecision(
                selected_fields_policy=SelectedFieldsPolicy.SOFT_FAIL,
                confidence=_decision_confidence(confidence, 0.7),
                reasons=[*reasons, "request_confidence_medium"],
            )

        explicit_query_signal = _has_explicit_query_signal(request.query)
        preferred_chart_types_present = bool(request.preferred_chart_types)

        if explicit_query_signal:
            reasons.append("explicit_query_signal")

        if preferred_chart_types_present:
            reasons.append("preferred_chart_types_present")

        if explicit_query_signal or preferred_chart_types_present:
            return GroundingPolicyDecision(
                selected_fields_policy=SelectedFieldsPolicy.STRICT,
                confidence=_decision_confidence(confidence, 0.9),
                reasons=[*reasons, "request_confidence_high"],
            )

        return GroundingPolicyDecision(
            selected_fields_policy=SelectedFieldsPolicy.SOFT_FAIL,
            confidence=_decision_confidence(confidence, 0.65),
            reasons=[*reasons, "no_explicit_query_or_chart_signal"],
        )


def resolve_grounding_policy(
    request: VisRAGRequest,
    *,
    request_confidence: float | None = None,
    ambiguity_notes: Iterable[str] | None = None,
) -> GroundingPolicyDecision:
    return GroundingPolicyResolver().resolve(
        request,
        request_confidence=request_confidence,
        ambiguity_notes=ambiguity_notes,
    )


def _parse_policy(value: object) -> SelectedFieldsPolicy | None:
    if value is None:
        return None

    try:
        return SelectedFieldsPolicy(str(value))
    except ValueError:
        return None


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = value.strip()

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _decision_confidence(request_confidence: float, rule_confidence: float) -> float:
    return round(_clamp((request_confidence + rule_confidence) / 2.0), 4)


def _prefix_values(prefix: str, values: list[str]) -> list[str]:
    return [f"{prefix}:{value}" for value in values]


def _only_identifier_like_fields(
    selected_fields: list[str],
    column_by_name: dict[str, VisRAGColumnProfile],
) -> bool:
    return all(_is_identifier_like(column_by_name[field]) for field in selected_fields)


def _is_identifier_like(column: VisRAGColumnProfile) -> bool:
    name = column.name.strip().lower()
    role = str(column.role or "").strip().lower()
    semantic_type = semantic_type_from_role_or_dtype(column.role, column.semantic_type, column.raw_dtype)

    if role in {"id", "identifier", "primary_key", "key", "uuid"}:
        return True

    if semantic_type in {"identifier", "id"}:
        return True

    return name == "id" or name.endswith("_id") or name.endswith(" id") or "uuid" in name


def _has_explicit_query_signal(query: str) -> bool:
    normalized = " ".join(query.lower().replace("_", " ").replace("-", " ").split())

    explicit_phrases = {
        " by ",
        " across ",
        " over time",
        " relationship between",
        " distribution of",
        " compare ",
        " comparing ",
        " trend ",
        " correlation ",
        " scatter ",
        " histogram",
        " line chart",
        " bar chart",
    }

    padded = f" {normalized} "
    return any(phrase in padded for phrase in explicit_phrases)
