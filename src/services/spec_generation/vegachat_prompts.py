from __future__ import annotations

import json
from typing import Any

from src.domain.models import DataPreparationResult, DataProfile, SpecGenerationRequest
from src.services.data.profile.data_profile_prompt_formatter import DataProfilePromptFormatter
from src.services.spec_generation.vegachat_contract import VEGA_LITE_SCHEMA_URL, vegachat_output_contract

_CODEGEN_PROMPT_CHAR_BUDGET = 12000
_PREVIOUS_RESPONSE_CHAR_BUDGET = 1200
_REQUEST_ANALYSIS_CHAR_BUDGET = 2200
_CHECKLIST_CHAR_BUDGET = 1400
_VALIDATION_FEEDBACK_CHAR_BUDGET = 1400
_SEMANTIC_FEEDBACK_CHAR_BUDGET = 1200
_VISRAG_CONTEXT_CHAR_BUDGET = 1500


def build_vegachat_codegen_prompt(
        request: SpecGenerationRequest,
        *,
        prompt_version: str,
        max_context_chars: int,
        include_visrag_context: bool = True,
        previous_error: str | None = None,
        previous_response: str | None = None,
        rag_prompt_top_k: int = 2,
) -> str:
    parts = [
        _system_contract(prompt_version),
        _dataset_contract(request.data_profile, request.prepared),
        _request_contract(request),
        _chartsquared_generation_contract(request),
        _validation_feedback_contract(request),
        _semantic_feedback_contract(request),
        _visrag_context(request.visrag, max_context_chars=max_context_chars)
        if include_visrag_context else "",
        _output_contract(),
    ]
    if previous_error:
        parts.append("Previous attempt failed. Correct the full response.\nError:\n" + previous_error)
    if previous_response:
        parts.append("Previous response:\n" + _truncate_text(previous_response, _PREVIOUS_RESPONSE_CHAR_BUDGET))
    return _truncate_text("\n\n".join(part.strip() for part in parts if part.strip()), _CODEGEN_PROMPT_CHAR_BUDGET)


def _system_contract(prompt_version: str) -> str:
    return f"""
Role: senior data-visualization engineer. Task: write one valid Vega-Lite v5 JSON spec for scientific data analysis.
Prompt version: {prompt_version}.

Output contract:
- Return only <explain>...</explain><json>...</json>.
- <explain>: one concise English sentence.
- <json>: exactly one Vega-Lite object; no wrapper keys; no markdown.
- Do not include data or datasets; runtime attaches data.url.
- Set $schema exactly to "{VEGA_LITE_SCHEMA_URL}".

Core generation rules:
1. Use only safe field names from the schema block; never invent fields.
2. Preserve requested fields on visible channels; tooltip-only evidence is not enough.
3. Use channel aggregate/bin/timeUnit/sort/stack instead of unnecessary transforms.
4. Show requested grouping through color, row, column, facet, shape, xOffset, or an axis.
5. Avoid shared raw axes for incompatible metrics; use normalized severity, independent panels, or separate views.
6. If overall_severity is available for top-problem requests, rank by it and keep raw metrics in tooltip/details.
7. For repeat/facet, keep Vega-Lite dynamic: repeat.field is substituted by Vega-Lite at render time. Do not replace repeat references with fixed fields inside the nested spec. Titles and headers must be informative, not "Repeated metrics", "Metric", or "Value" alone.
8. Do not force x-axis labelAngle; leave it unset unless labels demonstrably cannot fit.
"""


def _dataset_contract(data_profile: DataProfile | None, prepared: DataPreparationResult) -> str:
    lines = ["Dataset schema and safe field names:"]
    if data_profile is not None:
        lines.append(DataProfilePromptFormatter.for_chart_generation(data_profile, max_columns=18))
    else:
        for safe in prepared.safe_columns[:18]:
            original = prepared.reverse_column_name_map.get(safe, safe)
            lines.append(f"- original={original!r}; safe={safe!r}; type=unknown; role=unknown")
    if prepared.column_name_map:
        compact_mapping = {original: safe for original, safe in prepared.column_name_map.items() if original != safe}
        if compact_mapping:
            lines.append("Column mapping original -> safe:")
            lines.append(json.dumps(compact_mapping, ensure_ascii=False, separators=(",", ":")))
    derived_fields = [field for field in prepared.safe_columns if field not in set(prepared.reverse_column_name_map)]
    if derived_fields:
        lines.append("Prepared derived fields:")
        lines.extend(f"- {field!r}" for field in derived_fields[:18])
    return "\n".join(lines)


def _request_contract(request: SpecGenerationRequest) -> str:
    lines = ["User request:", request.query]
    analysis = request.query_request_analysis
    if analysis is not None:
        safe_selected = [_to_safe(field, request.prepared) for field in analysis.selected_fields]
        payload = {
            "analysis_task": analysis.analysis_task,
            "selected_safe_fields": safe_selected,
            "aggregation_plan": analysis.aggregation_plan,
            "metric_semantics": analysis.metric_semantics,
            "ranking_strategy": analysis.ranking_strategy,
            "scale_strategy": analysis.scale_strategy,
            "visual_constraints": analysis.visual_constraints[:8],
            "chart_answerability": analysis.chart_answerability,
        }
        lines.append("Query request analysis JSON:")
        lines.append(_truncate_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
            _REQUEST_ANALYSIS_CHAR_BUDGET,
        ))
    return "\n".join(lines)


def _chartsquared_generation_contract(request: SpecGenerationRequest) -> str:
    requirements = dict(getattr(request, "visual_judge_requirements", {}) or {})
    analysis = request.query_request_analysis
    payload = {
        "analysis_task": analysis.analysis_task if analysis is not None else None,
        "aggregation_plan": analysis.aggregation_plan if analysis is not None else {},
        "must_be_visible": requirements.get("must_be_visible", [])[:8],
        "acceptable_visual_encodings": requirements.get("acceptable_visual_encodings", {}),
        "critical_failures": requirements.get("critical_failures", [])[:8],
        "yes_no_questions": requirements.get("yes_no_questions", [])[:6],
    }
    if not any(payload.values()):
        return (
            "Pre-generation visual checklist: required fields and grouping must be visible; "
            "the chart family must fit the analytical task; tooltip-only meaning is insufficient."
        )
    return (
        "Pre-generation visual checklist JSON. Satisfy these visible criteria before retry loops:\n"
        + _truncate_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str), _CHECKLIST_CHAR_BUDGET)
    )


def _to_safe(field: str, prepared: DataPreparationResult) -> str:
    return prepared.column_name_map.get(field, field)


def _validation_feedback_contract(request: SpecGenerationRequest) -> str:
    if not request.previous_validation_errors and not request.previous_repair_hints and not request.previous_invalid_spec:
        return (
            f"Generation attempt: {request.generation_attempt_number} of {request.max_generation_attempts}.\n"
            "No previous validation errors for this attempt."
        )
    payload: dict[str, Any] = {
        "attempt": request.generation_attempt_number,
        "max_attempts": request.max_generation_attempts,
        "errors": request.previous_validation_errors,
        "repair_hints": request.previous_repair_hints,
        "invalid_spec": _strip_data(request.previous_invalid_spec or {}),
    }
    return (
        "Previous spec validation failed. Correct the exact technical errors:\n"
        + _truncate_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str), _VALIDATION_FEEDBACK_CHAR_BUDGET)
    )


def _semantic_feedback_contract(request: SpecGenerationRequest) -> str:
    if not request.previous_semantic_feedback and not request.previous_chart_facts:
        return "No previous semantic visual feedback for this attempt."
    payload: dict[str, Any] = {
        "attempt": request.generation_attempt_number,
        "max_attempts": request.max_generation_attempts,
        "semantic_feedback": request.previous_semantic_feedback,
        "chart_facts": request.previous_chart_facts[-3:],
    }
    return (
        "Previous rendered chart was technically valid but did not sufficiently answer the user request. Address this feedback:\n"
        + _truncate_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str), _SEMANTIC_FEEDBACK_CHAR_BUDGET)
    )


def _visrag_context(visrag, *, max_context_chars: int) -> str:
    if visrag is None or not getattr(visrag.generation_guidance, "has_guidance", False):
        return "No VisRAG guidance was retrieved for this generation run."
    text = visrag.generation_guidance.prompt_text.strip()
    if not text:
        return "No VisRAG guidance was retrieved for this generation run."
    header = "Retrieved VisRAG guidance: use as constraints only; do not copy external specs.\n"
    return _truncate_text(header + text, min(max_context_chars, _VISRAG_CONTEXT_CHAR_BUDGET))


def _strip_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_data(item) for key, item in value.items() if key not in {"data", "datasets"}}
    if isinstance(value, list):
        return [_strip_data(item) for item in value]
    return value


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    value = text or ""
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 24)].rstrip() + "\n[truncated]"


def _output_contract() -> str:
    return vegachat_output_contract()
