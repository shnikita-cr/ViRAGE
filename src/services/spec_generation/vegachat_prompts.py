from __future__ import annotations

import json
from typing import Any

from src.domain.models import DataPreparationResult, DataProfile, SpecGenerationRequest
from src.services.data_profile_prompt_formatter import DataProfilePromptFormatter

VEGA_LITE_SCHEMA_URL = "https://vega.github.io/schema/vega-lite/v5.json"


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
        parts.append(
            "Previous attempt failed. Generate a corrected full response.\n"
            f"Error:\n{previous_error}\n"
        )
    if previous_response:
        parts.append(f"Previous response:\n{previous_response[:4000]}\n")

    return "\n\n".join(part.strip() for part in parts if part.strip())


def _system_contract(prompt_version: str) -> str:
    return f"""
You are VegaLiteSpecCodegenAI.
Prompt version: {prompt_version}.
Generate one valid Vega-Lite v5 JSON specification for the provided dataset and user request.

Output rules:
1. Return only <explain>...</explain> and <json>...</json>.
2. The <explain> block must be concise English.
3. The <json> block must contain exactly one Vega-Lite object.
4. Do not include data or datasets. Runtime attaches data.url.
5. Set $schema exactly to "{VEGA_LITE_SCHEMA_URL}".

Generation rules adapted from VegaChat-style correction loops:
1. Use only safe field names listed in the schema block; never invent fields.
2. Use the requested fields before visually similar alternatives.
3. Choose the chart family from the analytic task: relationship -> point/scatter, trend -> line, comparison -> bar, distribution -> bin/histogram, part-to-whole -> stacked/normalized composition only when appropriate.
4. If the request says against/versus/relationship between two numeric fields, preserve both fields on visible quantitative channels.
5. If the request says split by, grouped by, broken down by, for each, or by category, make that grouping visible through color, row, column, facet, shape, or xOffset; tooltip-only grouping is not enough.
6. Prefer channel-level aggregate/bin/timeUnit/sort/stack over unnecessary transform objects.
7. Use row/column encoding for simple faceting; use view-level facet only when a full nested spec is required.
8. For grouped bars, prefer xOffset, column, or facet when side-by-side comparison is requested.
9. For temporal trends, use a temporal or ordered x-axis and a readable time unit when needed.
10. For high-cardinality categories, avoid unreadable color legends; prefer top-k, facet, horizontal bars, filtering, or larger layout.
11. Axis and legend titles must name the source field and aggregation/time unit when used.
12. Add informative tooltips, but do not rely on tooltip for required visual meaning.
13. If previous technical validation feedback is provided, fix those exact errors.
14. If previous PNG-only visual feedback is provided, change the visible chart so the missing requirement is visible.
"""


def _dataset_contract(data_profile: DataProfile | None, prepared: DataPreparationResult) -> str:
    lines = ["Dataset schema and safe field names:"]
    if data_profile is not None:
        lines.append(DataProfilePromptFormatter.for_chart_generation(data_profile))
    else:
        for safe in prepared.safe_columns:
            original = prepared.reverse_column_name_map.get(safe, safe)
            lines.append(f"- original={original!r}; safe={safe!r}; type=unknown; role=unknown")
    if prepared.column_name_map:
        compact_mapping = {original: safe for original, safe in prepared.column_name_map.items() if original != safe}
        if compact_mapping:
            lines.append("Column mapping original -> safe:")
            lines.append(json.dumps(compact_mapping, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def _request_contract(request: SpecGenerationRequest) -> str:
    lines = ["User request:", request.query]
    analysis = request.query_request_analysis
    if analysis is not None:
        safe_selected = [_to_safe(field, request.prepared) for field in analysis.selected_fields]
        payload = {
            "normalized_query": analysis.normalized_query,
            "analysis_task": analysis.analysis_task,
            "selected_original_fields": analysis.selected_fields,
            "selected_safe_fields": safe_selected,
            "field_bindings": {key: value.model_dump() for key, value in analysis.field_bindings.items()},
            "aggregation_plan": analysis.aggregation_plan,
            "chart_answerability": analysis.chart_answerability,
            "assumptions": analysis.assumptions,
            "ambiguity": analysis.ambiguity.model_dump(),
        }
        lines.append("Query request analysis:")
        lines.append(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return "\n".join(lines)


def _chartsquared_generation_contract(request: SpecGenerationRequest) -> str:
    """Add ChartSquared-style pre-generation criteria without adding a new runtime module."""
    requirements = dict(getattr(request, "visual_judge_requirements", {}) or {})
    analysis = request.query_request_analysis
    payload = {
        "analysis_task": analysis.analysis_task if analysis is not None else None,
        "aggregation_plan": analysis.aggregation_plan if analysis is not None else {},
        "chart_answerability": analysis.chart_answerability if analysis is not None else {},
        "must_be_visible": requirements.get("must_be_visible", []),
        "acceptable_visual_encodings": requirements.get("acceptable_visual_encodings", {}),
        "critical_failures_to_avoid": requirements.get("critical_failures", []),
        "yes_no_questions_to_satisfy_visually": requirements.get("yes_no_questions", [])[:8],
    }
    if not any(payload.values()):
        return (
            "ChartSquared-style pre-generation checklist:\n"
            "Before writing the JSON, derive a visual checklist from the user request and make the static chart "
            "satisfy it: required fields must be visible, required grouping must be visible, and the chosen chart "
            "family must match the analytical task. Do not hide required meaning only in tooltip."
        )
    return (
            "ChartSquared-style pre-generation checklist. The generated static PNG must satisfy these visible criteria; "
            "avoid every critical failure before relying on retry loops:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
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
        "generation_attempt_number": request.generation_attempt_number,
        "max_generation_attempts": request.max_generation_attempts,
        "previous_validation_errors": request.previous_validation_errors,
        "previous_repair_hints": request.previous_repair_hints,
        "previous_invalid_spec_without_large_data": _strip_data(request.previous_invalid_spec or {}),
    }
    return (
            "Previous spec validation failed. Generate a corrected Vega-Lite spec.\n"
            "Validation feedback for this retry:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def _semantic_feedback_contract(request: SpecGenerationRequest) -> str:
    if not request.previous_semantic_feedback and not request.previous_chart_facts:
        return "No previous semantic visual feedback for this attempt."

    payload: dict[str, Any] = {
        "generation_attempt_number": request.generation_attempt_number,
        "max_generation_attempts": request.max_generation_attempts,
        "previous_semantic_feedback": request.previous_semantic_feedback,
        "previous_chart_facts": request.previous_chart_facts[-3:],
    }
    return (
            "Previous rendered chart was technically valid but did not sufficiently answer the user request. "
            "Generate a new Vega-Lite spec that addresses the semantic feedback below.\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def _visrag_context(visrag, *, max_context_chars: int) -> str:
    if visrag is None or not getattr(visrag.generation_guidance, "has_guidance", False):
        return "No VisRAG guidance was retrieved for this generation run."
    text = visrag.generation_guidance.prompt_text.strip()
    if not text:
        return "No VisRAG guidance was retrieved for this generation run."
    header = (
        "Retrieved VisRAG guidance. Use it as rules and constraints only; "
        "do not copy any external Vega-Lite specification from RAG.\n"
    )
    payload = header + text
    if len(payload) > max_context_chars:
        return payload[:max_context_chars] + "\n[VisRAG guidance truncated]"
    return payload


def _strip_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_data(item) for key, item in value.items() if key not in {"data", "datasets"}}
    if isinstance(value, list):
        return [_strip_data(item) for item in value]
    return value

def _output_contract() -> str:
    return f"""
Return format:
<explain>
Short explanation in English.
</explain>

<json>
{{
  "$schema": "{VEGA_LITE_SCHEMA_URL}",
  "mark": "bar",
  "encoding": {{
    "x": {{"field": "safe_dimension_name", "type": "nominal"}},
    "y": {{"field": "safe_measure_name", "type": "quantitative", "aggregate": "mean"}}
  }}
}}
</json>

Layer/facet/repeat/concat specifications are also allowed when the user request requires them.
"""
