from __future__ import annotations

SUPPORTED_CORPUS_TYPES = {
    "example",
    "rule",
    "dataset_profile",
    "success_case",
    "failure_case",
    "utterance",
    "ambiguity_case",
    "design_constraint",
}

SUPPORTED_SEMANTIC_TYPES = {
    "quantitative",
    "temporal",
    "ordinal",
    "nominal",
    "geojson",
    "boolean",
    "unknown",
}

SUPPORTED_MARK_TYPES = {
    "area",
    "bar",
    "circle",
    "line",
    "point",
    "rect",
    "rule",
    "square",
    "text",
    "tick",
    "geoshape",
    "boxplot",
    "errorbar",
    "errorband",
}

SUPPORTED_CHANNELS = {
    "x",
    "y",
    "color",
    "size",
    "shape",
    "opacity",
    "row",
    "column",
    "theta",
    "radius",
    "detail",
    "tooltip",
    "longitude",
    "latitude",
    "text",
}


def validate_corpus_record_shape(record: dict) -> list[str]:
    errors: list[str] = []
    if not str(record.get("id") or "").strip():
        errors.append("missing_id")
    if str(record.get("corpus_type") or "") not in SUPPORTED_CORPUS_TYPES:
        errors.append("unsupported_corpus_type")
    if not str(record.get("instruction") or "").strip():
        errors.append("missing_instruction")
    mark_type = str(record.get("mark_type") or record.get("chart_type") or "")
    if mark_type and mark_type not in SUPPORTED_MARK_TYPES:
        errors.append("unsupported_mark_type")
    field_roles = record.get("field_roles") or {}
    if not isinstance(field_roles, dict):
        errors.append("field_roles_not_object")
    else:
        for channel, role in field_roles.items():
            if str(channel) not in SUPPORTED_CHANNELS:
                errors.append(f"unsupported_channel:{channel}")
            if str(role) not in SUPPORTED_SEMANTIC_TYPES:
                errors.append(f"unsupported_semantic_type:{role}")
    spec_template = record.get("spec_template")
    if spec_template is not None and not isinstance(spec_template, dict):
        errors.append("spec_template_not_object_or_null")
    return errors
