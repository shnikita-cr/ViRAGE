from __future__ import annotations

import json
from typing import Any

from src.domain.models import DataPreparationResult, DataProfile, SpecGenerationRequest
from src.llm.model_runtime import ModelRuntimeProfile
from src.llm.prompt_budget import PromptSection, build_budgeted_prompt, compact_text_to_tokens
from src.services.data.profile.data_profile_prompt_formatter import DataProfilePromptFormatter
from src.services.spec_generation.vegachat_contract import VEGA_LITE_SCHEMA_URL, vegachat_output_contract

_ALLOWED_CONSTANTS = {
    "mark.type": ["bar", "line", "point", "circle", "square", "area", "rect", "rule", "tick", "text", "boxplot", "errorbar"],
    "field.type": ["quantitative", "nominal", "ordinal", "temporal"],
    "aggregate": ["count", "mean", "median", "sum", "min", "max", "stdev"],
    "sort": ["ascending", "descending"],
    "scale.type": ["linear", "log", "sqrt", "time", "utc", "ordinal", "band"],
    "legend.orient": ["left", "right", "top", "bottom"],
}


def build_vegachat_codegen_prompt(
        request: SpecGenerationRequest,
        *,
        prompt_version: str,
        max_context_chars: int,
        include_visrag_context: bool = True,
        previous_error: str | None = None,
        previous_response: str | None = None,
        rag_prompt_top_k: int = 2,
        runtime_profile: ModelRuntimeProfile | None = None,
) -> str:
    profile_budget = runtime_profile.data_profile_budget_tokens if runtime_profile is not None else 1200
    rag_budget = runtime_profile.rag_budget_tokens if runtime_profile is not None else max(256, max_context_chars // 4)
    validation_budget = runtime_profile.validation_error_budget_tokens if runtime_profile is not None else 500
    sections = [
        PromptSection("system", _system_contract(prompt_version), priority=0),
        PromptSection("allowed_constants", _allowed_constants_contract(), priority=0),
        PromptSection("dataset", _dataset_contract(request.data_profile, request.prepared, profile_budget), min_tokens=256, priority=1),
        PromptSection("request", _request_contract(request), priority=0),
        PromptSection("chart_checklist", _chartsquared_generation_contract(request), min_tokens=160, priority=2),
        PromptSection("validation_feedback", _validation_feedback_contract(request, validation_budget), min_tokens=160, priority=1),
        PromptSection("semantic_feedback", _semantic_feedback_contract(request, validation_budget), min_tokens=160, priority=1),
        PromptSection("visrag", _visrag_context(request.visrag, max_context_chars=max_context_chars, token_budget=rag_budget) if include_visrag_context else "", min_tokens=160, priority=3),
        PromptSection("output", _output_contract(), priority=0),
    ]
    if previous_error:
        sections.append(PromptSection("previous_error", "Previous attempt failed. Generate a new full response.\nError:\n" + compact_text_to_tokens(previous_error, validation_budget), min_tokens=120, priority=1))
    if previous_response:
        sections.append(PromptSection("previous_response", "Previous response:\n" + compact_text_to_tokens(previous_response, validation_budget), min_tokens=120, priority=3))
    prompt, _ = build_budgeted_prompt(sections, profile=runtime_profile)
    return prompt


def _system_contract(prompt_version: str) -> str:
    return f"""
You are a data-visualization engineer. Write one valid Vega-Lite v5 JSON specification for a scientific chart.
Prompt version: {prompt_version}.
Output only <explain>...</explain><json>...</json>. No markdown, no code fences, no text outside tags.
The <json> block must contain one Vega-Lite object. Do not include data or datasets; runtime attaches data.url.
Set $schema exactly to "{VEGA_LITE_SCHEMA_URL}". Use only listed safe fields.

Hard design rules:
- Required fields, grouping, ranking, and comparison must be visible, not tooltip-only.
- Axis and legend titles must name the field plus aggregate or timeUnit when used.
- Prefer channel aggregate/bin/timeUnit/sort/stack over unnecessary transforms.
- Do not compare different-scale raw metrics on one shared quantitative axis; use normalized severity, independent facets, or separate views.
- If ranking_strategy uses severity and severity fields are available, rank by the severity field and keep raw metrics in tooltip/details.
- mark must be internally consistent: do not mix bar, boxplot, line, point, area, or errorbar-only properties in one mark object.
- For repeat/facet: Vega-Lite dynamically substitutes repeat.field in panel headers. Keep repeat references in encoding.field and use informative headers/titles; never use generic titles like "Repeated metrics", "Metric", or "Value".
- Do not force x-axis label rotation unless labels clearly cannot fit.
"""


def _allowed_constants_contract() -> str:
    return "Allowed common Vega-Lite constants:\n" + json.dumps(_ALLOWED_CONSTANTS, ensure_ascii=False, separators=(",", ":"))


def _dataset_contract(data_profile: DataProfile | None, prepared: DataPreparationResult, token_budget: int) -> str:
    lines = ["Dataset schema and safe field names:"]
    if data_profile is not None:
        lines.append(DataProfilePromptFormatter.for_chart_generation(data_profile, token_budget=token_budget))
    else:
        for safe in prepared.safe_columns:
            original = prepared.reverse_column_name_map.get(safe, safe)
            lines.append(f"- original={original!r}; safe={safe!r}; type=unknown; role=unknown")
    if prepared.column_name_map:
        compact_mapping = {original: safe for original, safe in prepared.column_name_map.items() if original != safe}
        if compact_mapping:
            lines.append("Column mapping original -> safe:")
            lines.append(json.dumps(compact_mapping, ensure_ascii=False, separators=(",", ":")))
    derived_fields = [field for field in prepared.safe_columns if field not in set(prepared.reverse_column_name_map)]
    if derived_fields:
        lines.append("Prepared derived fields available: " + ", ".join(derived_fields[:20]))
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
            "aggregation_plan": analysis.aggregation_plan,
            "metric_semantics": analysis.metric_semantics,
            "ranking_strategy": analysis.ranking_strategy,
            "scale_strategy": analysis.scale_strategy,
            "visual_constraints": analysis.visual_constraints,
            "chart_answerability": analysis.chart_answerability,
            "ambiguity": analysis.ambiguity.model_dump(),
        }
        lines.append("Query request analysis:")
        lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))
    return "\n".join(lines)


def _chartsquared_generation_contract(request: SpecGenerationRequest) -> str:
    requirements = dict(getattr(request, "visual_judge_requirements", {}) or {})
    analysis = request.query_request_analysis
    payload = {
        "analysis_task": analysis.analysis_task if analysis is not None else None,
        "aggregation_plan": analysis.aggregation_plan if analysis is not None else {},
        "must_be_visible": requirements.get("must_be_visible", []),
        "acceptable_visual_encodings": requirements.get("acceptable_visual_encodings", {}),
        "critical_failures_to_avoid": requirements.get("critical_failures", []),
        "yes_no_questions_to_satisfy_visually": requirements.get("yes_no_questions", [])[:8],
    }
    if not any(payload.values()):
        return "Chart checklist: required fields/grouping must be visible; chart family must match the analytical task; tooltip-only meaning is insufficient."
    return "ChartSquared-style pre-generation checklist. The generated static PNG must satisfy these visible criteria; avoid every critical failure before relying on retry loops.\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _to_safe(field: str, prepared: DataPreparationResult) -> str:
    return prepared.column_name_map.get(field, field)


def _validation_feedback_contract(request: SpecGenerationRequest, token_budget: int) -> str:
    if not request.previous_validation_errors and not request.previous_repair_hints and not request.previous_invalid_spec:
        return f"Generation attempt: {request.generation_attempt_number} of {request.max_generation_attempts}. No previous validation errors."
    payload = {
        "generation_attempt_number": request.generation_attempt_number,
        "max_generation_attempts": request.max_generation_attempts,
        "previous_validation_errors": request.previous_validation_errors,
        "previous_repair_hints": request.previous_repair_hints,
        "previous_invalid_spec_without_large_data": _strip_data(request.previous_invalid_spec or {}),
    }
    return "Previous spec validation failed. Generate a new valid Vega-Lite spec; do not patch the invalid object mechanically.\n" + compact_text_to_tokens(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str), token_budget)


def _semantic_feedback_contract(request: SpecGenerationRequest, token_budget: int) -> str:
    if not request.previous_semantic_feedback and not request.previous_chart_facts:
        return "No previous semantic visual feedback for this attempt."
    payload = {
        "generation_attempt_number": request.generation_attempt_number,
        "max_generation_attempts": request.max_generation_attempts,
        "previous_semantic_feedback": request.previous_semantic_feedback,
        "previous_chart_facts": request.previous_chart_facts[-3:],
    }
    return "Previous rendered chart was technically valid but did not sufficiently answer the user request. Generate a new spec that addresses this feedback.\n" + compact_text_to_tokens(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str), token_budget)


def _visrag_context(visrag, *, max_context_chars: int, token_budget: int) -> str:
    if visrag is None or not getattr(visrag.generation_guidance, "has_guidance", False):
        return "No VisRAG guidance was retrieved for this generation run."
    text = visrag.generation_guidance.prompt_text.strip()
    if not text:
        return "No VisRAG guidance was retrieved for this generation run."
    header = "Retrieved VisRAG guidance. Use it as rules only; do not copy external specs.\n"
    payload = header + text[:max_context_chars]
    return compact_text_to_tokens(payload, token_budget)


def _strip_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_data(item) for key, item in value.items() if key not in {"data", "datasets"}}
    if isinstance(value, list):
        return [_strip_data(item) for item in value]
    return value


def _output_contract() -> str:
    return vegachat_output_contract()
