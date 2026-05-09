from __future__ import annotations

import json
from typing import Any

from src.domain.models import CandidateSpecSet, DataPreparationResult, DataProfile, RequestAnalysisResult, SpecGenerationRequest

VEGA_LITE_SCHEMA_URL = "https://vega.github.io/schema/vega-lite/v5.json"


def build_vegachat_codegen_prompt(
    request: SpecGenerationRequest,
    *,
    prompt_version: str,
    max_context_chars: int,
    include_visrag_context: bool = True,
    previous_error: str | None = None,
    previous_response: str | None = None,
) -> str:
    parts = [
        _system_contract(prompt_version),
        _dataset_contract(request.data_profile, request.prepared),
        _request_contract(request),
        _visrag_context(request.candidate_spec_set, max_context_chars=max_context_chars) if include_visrag_context else "VisRAG context is disabled for this generation run.",
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
Your task is to generate a valid Vega-Lite v5 JSON specification for the dataset and user request.

Hard rules:
1. Return two XML-like blocks only: <explain>...</explain> and <json>...</json>.
2. The <explain> block must be concise English.
3. The <json> block must contain one Vega-Lite JSON object.
4. The model output JSON must not contain data or datasets. Runtime will attach data on the server.
5. The $schema field must be exactly "{VEGA_LITE_SCHEMA_URL}".
6. Use only safe Vega-Lite field names listed below.
7. Do not invent fields.
8. Do not copy axis titles, legend titles, scale domains, or sort arrays from retrieved examples unless they directly match current dataset fields.
9. Prefer channel-level aggregate/bin/timeUnit/sort/stack over view-level transform when possible.
10. If faceting is needed, prefer row/column encoding channels over the facet view-level operator.
"""


def _dataset_contract(data_profile: DataProfile | None, prepared: DataPreparationResult) -> str:
    lines = ["Dataset schema and safe field names:"]
    if data_profile is None:
        for safe in prepared.safe_columns:
            original = prepared.reverse_column_name_map.get(safe, safe)
            lines.append(f"- original: {original!r}; safe: {safe!r}; type: unknown; role: unknown")
    else:
        by_original = {column.original_name or column.name: column for column in data_profile.columns}
        for safe in prepared.safe_columns:
            original = prepared.reverse_column_name_map.get(safe, safe)
            column = by_original.get(original)
            dtype = column.dtype if column is not None else "unknown"
            role = data_profile.field_roles.get(original, "unknown")
            missing = column.missing_ratio if column is not None else None
            unique = column.unique_count if column is not None else None
            stats = []
            if missing is not None:
                stats.append(f"missing_ratio={missing:.3f}")
            if unique is not None:
                stats.append(f"unique_count={unique}")
            if column is not None and column.min_value is not None:
                stats.append(f"min={column.min_value}")
            if column is not None and column.max_value is not None:
                stats.append(f"max={column.max_value}")
            stat_text = "; ".join(stats)
            lines.append(f"- original: {original!r}; safe: {safe!r}; type: {dtype}; role: {role}; {stat_text}")
        lines.append(f"Rows: {data_profile.row_count}; columns: {data_profile.col_count}; complexity: {data_profile.data_complexity or 'unknown'}")
        if data_profile.quality_notes:
            lines.append("Quality notes:")
            for note in data_profile.quality_notes[:8]:
                lines.append(f"- {note}")

    if prepared.column_name_map:
        lines.append("Column mapping original -> safe:")
        lines.append(json.dumps(prepared.column_name_map, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def _request_contract(request: SpecGenerationRequest) -> str:
    lines = [
        "User request:",
        request.query,
    ]
    if request.query_understanding is not None:
        lines.append("Query understanding:")
        lines.append(json.dumps(request.query_understanding.model_dump(), ensure_ascii=False, indent=2))
    if request.request_analysis is not None:
        safe_selected = [_to_safe(field, request.prepared) for field in request.request_analysis.selected_fields]
        lines.append("Request analysis selected fields:")
        lines.append(json.dumps(
            {
                "selected_original_fields": request.request_analysis.selected_fields,
                "selected_safe_fields": safe_selected,
                "grounded_fields": request.request_analysis.grounded_fields,
                "missing_fields": request.request_analysis.missing_fields,
                "confidence": request.request_analysis.confidence,
            },
            ensure_ascii=False,
            indent=2,
        ))
    return "\n".join(lines)


def _to_safe(field: str, prepared: DataPreparationResult) -> str:
    return prepared.column_name_map.get(field, field)


def _visrag_context(candidate_spec_set: CandidateSpecSet, *, max_context_chars: int) -> str:
    payload: dict[str, Any] = {
        "selected_candidate": None,
        "candidate_specs": [],
        "retrieved_examples": [],
        "ranking_hints": candidate_spec_set.ranking_hints[:10],
    }

    selected = candidate_spec_set.selected_candidate_spec
    if selected is not None:
        payload["selected_candidate"] = _candidate_payload(selected)

    for candidate in candidate_spec_set.candidate_specs[:5]:
        payload["candidate_specs"].append(_candidate_payload(candidate))

    for example in candidate_spec_set.retrieved_examples[:5]:
        payload["retrieved_examples"].append(
            {
                "example_id": example.example_id,
                "chart_type": example.chart_type,
                "instruction": example.instruction,
                "description": example.description,
                "score": example.score,
                "rationale": example.rationale,
                "tags": example.tags[:8],
            }
        )

    text = (
        "Retrieved VisRAG context. Use this as guidance, not as a template to copy blindly.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )
    if len(text) > max_context_chars:
        return text[:max_context_chars] + "\n[VisRAG context truncated]"
    return text


def _candidate_payload(candidate) -> dict[str, Any]:
    spec_template = candidate.spec_template or {}
    compact_template = _strip_data(spec_template)
    return {
        "spec_id": candidate.spec_id,
        "chart_family": candidate.chart_family,
        "summary": candidate.summary,
        "score": candidate.score,
        "rationale": candidate.rationale,
        "encoding_roles": candidate.encoding_roles,
        "transform_types": candidate.transform_types,
        "field_mapping": candidate.field_mapping,
        "spec_template_without_data": compact_template,
    }


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
  "mark": "...",
  "encoding": {{}}
}}
</json>
"""
