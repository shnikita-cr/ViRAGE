from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_INPUT_JSONL = "rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl"
DEFAULT_OUTPUT_JSONL = "rag_corpus/data/vega_lite_examples.jsonl"
DEFAULT_REPORT_MD = "rag_corpus/reports/visrag_runtime_export_report.md"
DEFAULT_REPORT_JSON = "rag_corpus/reports/visrag_runtime_export_report.json"

DATA_KEYS_TO_REMOVE = {"data", "datasets"}

COMPLEX_TOP_LEVEL_KEYS = {
    "layer",
    "facet",
    "repeat",
    "concat",
    "hconcat",
    "vconcat",
}

SUPPORTED_RUNTIME_CHART_TYPES = {
    "bar",
    "line",
    "area",
    "point",
    "circle",
    "square",
    "tick",
    "histogram",
    "boxplot",
    "rect",
    "rule",
    "text",
}

CHART_PATTERN_TO_RUNTIME_CHART_TYPE = {
    "bar_chart": "bar",
    "line_chart": "line",
    "scatter_plot": "point",
    "histogram": "histogram",
    "heatmap": "rect",
    "area_chart": "area",
    "point_chart": "point",
    "tick_plot": "tick",
    "rule_chart": "rule",
    "text_chart": "text",
    "rect_chart": "rect",
    # These are intentionally unsupported by current ViRAGE chart_types.py.
    "pie_chart": "arc",
    "map_chart": "geoshape",
    # Complex layouts are skipped by default.
    "layered_chart": "layer",
    "faceted_chart": "facet",
}

MARK_TYPE_TO_RUNTIME_CHART_TYPE = {
    "bar": "bar",
    "line": "line",
    "area": "area",
    "point": "point",
    "circle": "circle",
    "square": "square",
    "tick": "tick",
    "rect": "rect",
    "rule": "rule",
    "text": "text",
    "boxplot": "boxplot",
}

ALLOWED_RUNTIME_CHANNELS = {
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
    "longitude",
    "latitude",
    "text",
}

DROPPED_ENCODING_CHANNELS_BY_DEFAULT = {
    "tooltip",
    "href",
    "url",
    "key",
    "order",
}

BASE_SEMANTIC_TYPES = {
    "quantitative",
    "temporal",
    "nominal",
    "ordinal",
    "boolean",
    "geojson",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL file does not exist: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc

            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at line {line_number}")

            records.append(value)

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def remove_data_sections(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: remove_data_sections(child)
            for key, child in value.items()
            if key not in DATA_KEYS_TO_REMOVE
        }

    if isinstance(value, list):
        return [remove_data_sections(item) for item in value]

    return value


def has_complex_top_level_structure(spec: dict[str, Any]) -> bool:
    return any(key in spec for key in COMPLEX_TOP_LEVEL_KEYS)


def has_transforms(spec: dict[str, Any]) -> bool:
    value = spec.get("transform")
    return isinstance(value, list) and len(value) > 0


def has_interactive_params(spec: dict[str, Any]) -> bool:
    if "selection" in spec:
        return True

    params = spec.get("params")

    if not isinstance(params, list):
        return False

    for item in params:
        if not isinstance(item, dict):
            continue

        if "select" in item or "bind" in item:
            return True

    return False


def remove_interaction_sections(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}

        for key, child in value.items():
            if key in {"params", "selection"}:
                continue

            if key == "condition":
                # Conditional encodings usually refer to a param/selection.
                # They are unsafe if params are removed, so we drop the condition.
                continue

            cleaned[key] = remove_interaction_sections(child)

        return cleaned

    if isinstance(value, list):
        return [remove_interaction_sections(item) for item in value]

    return value


def get_mark_type_from_spec(spec: dict[str, Any]) -> str | None:
    mark = spec.get("mark")

    if isinstance(mark, str):
        return mark

    if isinstance(mark, dict):
        mark_type = mark.get("type")

        if isinstance(mark_type, str):
            return mark_type

    return None


def canonicalize_runtime_chart_type(value: str | None) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower().replace("-", "_")

    aliases = {
        "scatter": "point",
        "scatterplot": "point",
        "scatter_plot": "point",
        "scatter plot": "point",
        "bar chart": "bar",
        "bar_chart": "bar",
        "column": "bar",
        "column_chart": "bar",
        "line chart": "line",
        "line_chart": "line",
        "area chart": "area",
        "area_chart": "area",
        "hist": "histogram",
        "histogram chart": "histogram",
        "box": "boxplot",
        "box_plot": "boxplot",
        "heatmap": "rect",
        "heat_map": "rect",
        "matshow": "rect",
        "imshow": "rect",
    }

    return aliases.get(text, text)


def choose_runtime_chart_type(record: dict[str, Any], spec_template: dict[str, Any]) -> tuple[str, str]:
    chart_pattern = record.get("chart_pattern")
    mark_type = record.get("mark_type")

    if isinstance(chart_pattern, str):
        mapped = CHART_PATTERN_TO_RUNTIME_CHART_TYPE.get(chart_pattern.strip().lower())

        if mapped:
            return canonicalize_runtime_chart_type(mapped), "chart_pattern"

    if isinstance(mark_type, str):
        mapped = MARK_TYPE_TO_RUNTIME_CHART_TYPE.get(mark_type.strip().lower())

        if mapped:
            return canonicalize_runtime_chart_type(mapped), "mark_type"

    spec_mark_type = get_mark_type_from_spec(spec_template)
    mapped = MARK_TYPE_TO_RUNTIME_CHART_TYPE.get(str(spec_mark_type or "").strip().lower())

    if mapped:
        return canonicalize_runtime_chart_type(mapped), "spec_mark"

    return "", "unresolved"


def simplify_semantic_role(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()

    if not text:
        return None

    # Count is a valid Vega-Lite aggregate, but not a user dataframe field role.
    if text in {"count", "count_aggregate"}:
        return None

    if text.startswith("quantitative"):
        return "quantitative"

    if text.startswith("temporal"):
        return "temporal"

    if text.startswith("nominal"):
        return "nominal"

    if text.startswith("ordinal"):
        return "ordinal"

    if text.startswith("boolean"):
        return "boolean"

    if text.startswith("geojson") or text.startswith("geo"):
        return "geojson"

    if text in {"number", "numeric", "measure", "int", "integer", "float", "double", "decimal"}:
        return "quantitative"

    if text in {"date", "time", "datetime", "timestamp", "year", "month"}:
        return "temporal"

    if text in {"string", "str", "object", "category", "categorical", "dimension"}:
        return "nominal"

    if text in BASE_SEMANTIC_TYPES:
        return text

    return None


def simplify_field_roles(field_roles: Any) -> tuple[dict[str, str], dict[str, str]]:
    if not isinstance(field_roles, dict):
        return {}, {"__all__": "field_roles is not an object"}

    simplified: dict[str, str] = {}
    dropped: dict[str, str] = {}

    for channel, role in field_roles.items():
        channel_name = str(channel)

        if channel_name not in ALLOWED_RUNTIME_CHANNELS:
            dropped[channel_name] = "unsupported_channel"
            continue

        simplified_role = simplify_semantic_role(role)

        if not simplified_role:
            dropped[channel_name] = f"unsupported_or_non_field_role:{role}"
            continue

        simplified[channel_name] = simplified_role

    return simplified, dropped


def infer_semantic_role_from_channel_def(channel_def: Any) -> str | None:
    if isinstance(channel_def, list):
        for item in channel_def:
            inferred = infer_semantic_role_from_channel_def(item)
            if inferred:
                return inferred

        return None

    if not isinstance(channel_def, dict):
        return None

    aggregate = channel_def.get("aggregate")
    field = channel_def.get("field")
    field_type = channel_def.get("type")
    has_bin = bool(channel_def.get("bin"))
    has_time_unit = bool(channel_def.get("timeUnit"))

    if aggregate == "count":
        return None

    if isinstance(field_type, str):
        simplified = simplify_semantic_role(field_type)

        if simplified:
            return simplified

    if has_time_unit:
        return "temporal"

    if has_bin:
        return "quantitative"

    if isinstance(aggregate, str):
        return "quantitative"

    # If there is a field but Vega-Lite omitted type, using nominal is safer than
    # dropping the role entirely for runtime smoke tests.
    if isinstance(field, str) and field.strip():
        return "nominal"

    return None


def infer_field_roles_from_spec_template(
        spec_template: dict[str, Any],
        keep_tooltip: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    encoding = spec_template.get("encoding")

    if not isinstance(encoding, dict):
        return {}, {"__all__": "spec_template.encoding is not an object"}

    inferred: dict[str, str] = {}
    dropped: dict[str, str] = {}

    for channel, channel_def in encoding.items():
        channel_name = str(channel)

        if channel_name in DROPPED_ENCODING_CHANNELS_BY_DEFAULT and not (
                channel_name == "tooltip" and keep_tooltip
        ):
            dropped[channel_name] = "dropped_encoding_channel"
            continue

        if channel_name not in ALLOWED_RUNTIME_CHANNELS:
            dropped[channel_name] = "unsupported_channel"
            continue

        role = infer_semantic_role_from_channel_def(channel_def)

        if not role:
            dropped[channel_name] = "no_runtime_field_role_in_encoding"
            continue

        inferred[channel_name] = role

    return inferred, dropped


def merge_field_roles(
        primary_roles: dict[str, str],
        primary_dropped: dict[str, str],
        fallback_roles: dict[str, str],
        fallback_dropped: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], str]:
    if not primary_roles and fallback_roles:
        return fallback_roles, {**primary_dropped, **fallback_dropped}, "spec_template_encoding"

    merged = dict(primary_roles)
    source = "normalized_field_roles"

    for channel, role in fallback_roles.items():
        if channel not in merged:
            merged[channel] = role
            source = "normalized_field_roles_plus_spec_template_encoding"

    return merged, {**primary_dropped, **fallback_dropped}, source


def strip_field_keys(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}

        for key, child in value.items():
            if key == "field":
                continue

            result[key] = strip_field_keys(child)

        return result

    if isinstance(value, list):
        return [strip_field_keys(item) for item in value]

    return value


def clean_encoding(
        encoding: Any,
        runtime_field_roles: dict[str, str],
        keep_tooltip: bool,
) -> Any:
    if not isinstance(encoding, dict):
        return encoding

    cleaned: dict[str, Any] = {}

    for channel, channel_spec in encoding.items():
        channel_name = str(channel)

        if channel_name in DROPPED_ENCODING_CHANNELS_BY_DEFAULT and not (
                channel_name == "tooltip" and keep_tooltip
        ):
            continue

        cleaned_channel_spec = strip_field_keys(channel_spec)

        # Keep channels that require runtime field mapping.
        if channel_name in runtime_field_roles:
            cleaned[channel_name] = cleaned_channel_spec
            continue

        # Keep count aggregate channels: Vega-Lite allows aggregate=count without a field.
        if _is_count_aggregate_channel(cleaned_channel_spec):
            cleaned[channel_name] = cleaned_channel_spec
            continue

        # Keep non-field constants/encodings only if they still have meaningful spec.
        if not _is_empty_or_fieldless_encoding(cleaned_channel_spec):
            cleaned[channel_name] = cleaned_channel_spec

    return cleaned


def _is_count_aggregate_channel(value: Any) -> bool:
    if isinstance(value, dict):
        return value.get("aggregate") == "count"

    return False


def _is_empty_or_fieldless_encoding(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, dict):
        meaningful_keys = set(value.keys()) - {"title"}
        return not meaningful_keys

    if isinstance(value, list):
        return all(_is_empty_or_fieldless_encoding(item) for item in value)

    return False


def sanitize_spec_template(
        spec_template: dict[str, Any],
        runtime_field_roles: dict[str, str],
        keep_tooltip: bool,
        strip_fields: bool,
        strip_interactions: bool,
) -> dict[str, Any]:
    spec = remove_data_sections(copy.deepcopy(spec_template))

    if strip_interactions:
        spec = remove_interaction_sections(spec)

    if strip_fields:
        spec = strip_field_keys(spec)

    encoding = spec.get("encoding")

    if isinstance(encoding, dict):
        spec["encoding"] = clean_encoding(
            encoding=encoding,
            runtime_field_roles=runtime_field_roles,
            keep_tooltip=keep_tooltip,
        )

    return spec


def extract_transform_types(spec: dict[str, Any]) -> list[str]:
    transforms = spec.get("transform")

    if not isinstance(transforms, list):
        return []

    result: list[str] = []

    for item in transforms:
        if not isinstance(item, dict):
            continue

        for key in item.keys():
            result.append(str(key))

    return sorted(set(result))


def split_title_terms(value: str) -> list[str]:
    terms = []

    for token in value.replace("-", "_").split("_"):
        token = token.strip().lower()

        if token:
            terms.append(token)

    return terms


def build_keywords(record: dict[str, Any], runtime_chart_type: str) -> list[str]:
    values: list[str] = []

    for field in ("title", "file_stem", "chart_pattern", "mark_type", "source", "corpus_type"):
        value = record.get(field)

        if isinstance(value, str) and value.strip():
            values.append(value.strip())

    if runtime_chart_type:
        values.append(runtime_chart_type)

    file_stem = record.get("file_stem")

    if isinstance(file_stem, str):
        values.extend(split_title_terms(file_stem))

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = str(value).strip().lower()

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result


def format_field_roles(field_roles: dict[str, str]) -> str:
    if not field_roles:
        return "none"

    return ", ".join(
        f"{channel} {role}"
        for channel, role in sorted(field_roles.items())
    )


def build_runtime_instruction(
        record: dict[str, Any],
        runtime_chart_type: str,
        runtime_field_roles: dict[str, str],
) -> str:
    title = record.get("title")
    instruction = record.get("instruction")
    chart_pattern = record.get("chart_pattern")
    mark_type = record.get("mark_type")

    parts: list[str] = []

    if isinstance(title, str) and title.strip():
        parts.append(f"Title: {title.strip()}.")

    if isinstance(instruction, str) and instruction.strip():
        parts.append(f"Instruction: {instruction.strip()}")

    if isinstance(chart_pattern, str) and chart_pattern.strip():
        parts.append(f"Chart pattern: {chart_pattern.strip()}.")

    if runtime_chart_type:
        parts.append(f"Runtime chart type: {runtime_chart_type}.")

    if isinstance(mark_type, str) and mark_type.strip():
        parts.append(f"Original mark type: {mark_type.strip()}.")

    parts.append(f"Field roles: {format_field_roles(runtime_field_roles)}.")
    parts.append("Source: Vega-Lite official example.")

    return " ".join(parts)


def build_metadata(
        record: dict[str, Any],
        runtime_chart_type: str,
        chart_type_source: str,
        field_roles_source: str,
        dropped_field_roles: dict[str, str],
        transform_types: list[str],
        had_interactive_params: bool,
) -> dict[str, Any]:
    metadata_fields = [
        "id",
        "source_path",
        "file_name",
        "file_stem",
        "title",
        "chart_pattern",
        "mark_type",
        "data_policy",
        "source_split",
        "benchmark_group",
        "is_eval_leak_sensitive",
        "removed_data_sections",
    ]

    metadata: dict[str, Any] = {
        "original_id": record.get("id"),
        "runtime_chart_type": runtime_chart_type,
        "chart_type_source": chart_type_source,
        "field_roles_source": field_roles_source,
        "dropped_field_roles": dropped_field_roles,
        "transform_types": transform_types,
        "had_interactive_params": had_interactive_params,
    }

    for field in metadata_fields:
        if field not in record:
            continue

        value = record[field]

        if isinstance(value, (dict, list)):
            metadata[field] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        elif value is None or isinstance(value, (str, int, float, bool)):
            metadata[field] = value
        else:
            metadata[field] = str(value)

    return metadata


def build_runtime_record(
        record: dict[str, Any],
        corpus_name: str,
        keep_tooltip: bool,
        strip_fields: bool,
        allow_interactive_params: bool,
        strip_interactions: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    original_id = record.get("id")

    if not isinstance(original_id, str) or not original_id.strip():
        return None, "missing_id"

    raw_spec_template = record.get("spec_template")

    if not isinstance(raw_spec_template, dict) or not raw_spec_template:
        return None, "missing_or_invalid_spec_template"

    if has_complex_top_level_structure(raw_spec_template):
        return None, "complex_top_level_structure"

    if has_transforms(raw_spec_template):
        return None, "transform_not_runtime_safe"

    had_interactive_params = has_interactive_params(raw_spec_template)

    if had_interactive_params and not allow_interactive_params:
        return None, "interactive_params_not_runtime_safe"

    runtime_chart_type, chart_type_source = choose_runtime_chart_type(
        record=record,
        spec_template=raw_spec_template,
    )

    if runtime_chart_type not in SUPPORTED_RUNTIME_CHART_TYPES:
        return None, f"unsupported_runtime_chart_type:{runtime_chart_type or 'empty'}"

    normalized_roles, normalized_dropped = simplify_field_roles(record.get("field_roles"))
    inferred_roles, inferred_dropped = infer_field_roles_from_spec_template(
        raw_spec_template,
        keep_tooltip=keep_tooltip,
    )
    runtime_field_roles, dropped_field_roles, field_roles_source = merge_field_roles(
        primary_roles=normalized_roles,
        primary_dropped=normalized_dropped,
        fallback_roles=inferred_roles,
        fallback_dropped=inferred_dropped,
    )

    if not runtime_field_roles:
        return None, "empty_runtime_field_roles"

    spec_template = sanitize_spec_template(
        spec_template=raw_spec_template,
        runtime_field_roles=runtime_field_roles,
        keep_tooltip=keep_tooltip,
        strip_fields=strip_fields,
        strip_interactions=strip_interactions,
    )

    if not isinstance(spec_template.get("encoding"), dict) or not spec_template["encoding"]:
        return None, "missing_or_empty_runtime_encoding"

    source = record.get("source")

    if not isinstance(source, str) or not source.strip():
        source = "unknown"

    transform_types = extract_transform_types(raw_spec_template)
    instruction = build_runtime_instruction(
        record=record,
        runtime_chart_type=runtime_chart_type,
        runtime_field_roles=runtime_field_roles,
    )

    description = record.get("instruction")

    if not isinstance(description, str) or not description.strip():
        description = record.get("title") if isinstance(record.get("title"), str) else None

    runtime_record = {
        "id": original_id,
        "source": source,
        "corpus": corpus_name,
        "corpus_type": "example",
        "instruction": instruction,
        "description": description.strip() if isinstance(description, str) else None,
        "keywords": build_keywords(record, runtime_chart_type),
        "chart_type": runtime_chart_type,
        "field_roles": runtime_field_roles,
        "transform_types": transform_types,
        "spec_template": spec_template,
        "metadata": build_metadata(
            record=record,
            runtime_chart_type=runtime_chart_type,
            chart_type_source=chart_type_source,
            field_roles_source=field_roles_source,
            dropped_field_roles=dropped_field_roles,
            transform_types=transform_types,
            had_interactive_params=had_interactive_params,
        ),
    }

    return runtime_record, None


def export_records(
        records: list[dict[str, Any]],
        corpus_name: str,
        keep_tooltip: bool,
        strip_fields: bool,
        allow_interactive_params: bool,
        strip_interactions: bool,
        max_records: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        runtime_record, skip_reason = build_runtime_record(
            record=record,
            corpus_name=corpus_name,
            keep_tooltip=keep_tooltip,
            strip_fields=strip_fields,
            allow_interactive_params=allow_interactive_params,
            strip_interactions=strip_interactions,
        )

        if skip_reason:
            skipped.append(
                {
                    "index": index,
                    "id": record.get("id"),
                    "title": record.get("title"),
                    "chart_pattern": record.get("chart_pattern"),
                    "mark_type": record.get("mark_type"),
                    "reason": skip_reason,
                }
            )
            continue

        included.append(runtime_record)

        if max_records is not None and len(included) >= max_records:
            break

    return included, skipped


def count_field(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()

    for record in records:
        value = record.get(field)

        if isinstance(value, str):
            counter[value] += 1
        elif value is None:
            counter["[null]"] += 1
        else:
            counter[str(value)] += 1

    return dict(counter.most_common())


def count_metadata_field(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()

    for record in records:
        metadata = record.get("metadata", {})

        if not isinstance(metadata, dict):
            counter["[invalid_metadata]"] += 1
            continue

        value = metadata.get(field)

        if isinstance(value, str):
            counter[value] += 1
        elif value is None:
            counter["[null]"] += 1
        else:
            counter[str(value)] += 1

    return dict(counter.most_common())


def build_report_data(
        input_jsonl: Path,
        output_jsonl: Path,
        records: list[dict[str, Any]],
        included: list[dict[str, Any]],
        skipped: list[dict[str, Any]],
        corpus_name: str,
        keep_tooltip: bool,
        strip_fields: bool,
        allow_interactive_params: bool,
        strip_interactions: bool,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": input_jsonl.as_posix(),
        "output_jsonl": output_jsonl.as_posix(),
        "corpus_name": corpus_name,
        "keep_tooltip": keep_tooltip,
        "strip_fields": strip_fields,
        "allow_interactive_params": allow_interactive_params,
        "strip_interactions": strip_interactions,
        "input_records": len(records),
        "included_records": len(included),
        "skipped_records": len(skipped),
        "included_chart_type_distribution": count_field(included, "chart_type"),
        "included_source_distribution": count_field(included, "source"),
        "included_original_chart_pattern_distribution": count_metadata_field(included, "chart_pattern"),
        "field_roles_source_distribution": count_metadata_field(included, "field_roles_source"),
        "had_interactive_params_distribution": count_metadata_field(included, "had_interactive_params"),
        "skip_reason_distribution": dict(Counter(item["reason"] for item in skipped).most_common()),
        "sample_included": included[:10],
        "sample_skipped": skipped[:50],
    }


def markdown_counter(title: str, values: dict[str, int]) -> list[str]:
    lines: list[str] = []

    lines.append(f"## {title}")
    lines.append("")

    if not values:
        lines.append("_No data._")
        lines.append("")
        return lines

    lines.append("| Value | Count |")
    lines.append("|---|---:|")

    for value, count in values.items():
        lines.append(f"| `{value}` | {count} |")

    lines.append("")
    return lines


def build_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# VisRAG runtime corpus export report")
    lines.append("")
    lines.append(f"Generated at: `{report['generated_at']}`")
    lines.append(f"Input JSONL: `{report['input_jsonl']}`")
    lines.append(f"Output JSONL: `{report['output_jsonl']}`")
    lines.append(f"Corpus name: `{report['corpus_name']}`")
    lines.append(f"Keep tooltip: `{report['keep_tooltip']}`")
    lines.append(f"Strip fields: `{report['strip_fields']}`")
    lines.append(f"Allow interactive params: `{report['allow_interactive_params']}`")
    lines.append(f"Strip interactions: `{report['strip_interactions']}`")
    lines.append(f"Input records: **{report['input_records']}**")
    lines.append(f"Included records: **{report['included_records']}**")
    lines.append(f"Skipped records: **{report['skipped_records']}**")
    lines.append("")

    lines.extend(
        markdown_counter(
            "Included runtime chart_type distribution",
            report["included_chart_type_distribution"],
        )
    )
    lines.extend(
        markdown_counter(
            "Included source distribution",
            report["included_source_distribution"],
        )
    )
    lines.extend(
        markdown_counter(
            "Included original chart_pattern distribution",
            report["included_original_chart_pattern_distribution"],
        )
    )
    lines.extend(
        markdown_counter(
            "Field roles source distribution",
            report["field_roles_source_distribution"],
        )
    )
    lines.extend(
        markdown_counter(
            "Included interactive params distribution",
            report["had_interactive_params_distribution"],
        )
    )
    lines.extend(
        markdown_counter(
            "Skip reason distribution",
            report["skip_reason_distribution"],
        )
    )

    lines.append("## Sample included records")
    lines.append("")
    lines.append("| id | chart_type | field_roles | field_roles_source | instruction preview |")
    lines.append("|---|---|---|---|---|")

    for record in report["sample_included"]:
        preview = str(record.get("instruction", ""))[:180].replace("|", "\\|").replace("\n", " ")
        field_roles = json.dumps(record.get("field_roles", {}), ensure_ascii=False)
        metadata = record.get("metadata", {})
        field_roles_source = metadata.get("field_roles_source") if isinstance(metadata, dict) else None
        lines.append(
            f"| `{record.get('id')}` | "
            f"`{record.get('chart_type')}` | "
            f"`{field_roles}` | "
            f"`{field_roles_source}` | "
            f"{preview} |"
        )

    lines.append("")

    lines.append("## Sample skipped records")
    lines.append("")
    lines.append("| id | title | chart_pattern | mark_type | reason |")
    lines.append("|---|---|---|---|---|")

    for item in report["sample_skipped"]:
        lines.append(
            f"| `{item.get('id')}` | "
            f"`{item.get('title')}` | "
            f"`{item.get('chart_pattern')}` | "
            f"`{item.get('mark_type')}` | "
            f"`{item.get('reason')}` |"
        )

    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], report_md: Path, report_json: Path) -> None:
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    report_md.write_text(build_markdown_report(report), encoding="utf-8")
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export normalized JSONL records to the runtime JSONL format expected "
            "by src/visrag_core/corpus.py."
        )
    )

    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Path to normalized source-of-truth JSONL.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Output runtime JSONL path for ViRAGE VisRAG.",
    )
    parser.add_argument(
        "--report-md",
        default=DEFAULT_REPORT_MD,
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--report-json",
        default=DEFAULT_REPORT_JSON,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--corpus-name",
        default="vega_lite",
        help="Runtime corpus name written to each record.",
    )
    parser.add_argument(
        "--keep-tooltip",
        action="store_true",
        help="Keep tooltip encoding channels. Default drops tooltip for runtime safety.",
    )
    parser.add_argument(
        "--keep-fields",
        action="store_true",
        help=(
            "Keep original field names in spec_template. "
            "Default removes field keys so runtime grounding inserts real dataframe fields."
        ),
    )
    parser.add_argument(
        "--allow-interactive-params",
        action="store_true",
        help=(
            "Allow specs with top-level selection or params select/bind. "
            "Default skips them for MVP runtime safety."
        ),
    )
    parser.add_argument(
        "--keep-interactions",
        action="store_true",
        help=(
            "Keep params/selection/condition sections in spec_template. "
            "Default strips them when a spec is included."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional maximum number of included runtime records.",
    )

    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    report_md = Path(args.report_md)
    report_json = Path(args.report_json)

    records = load_jsonl(input_jsonl)
    included, skipped = export_records(
        records=records,
        corpus_name=args.corpus_name,
        keep_tooltip=args.keep_tooltip,
        strip_fields=not args.keep_fields,
        allow_interactive_params=args.allow_interactive_params,
        strip_interactions=not args.keep_interactions,
        max_records=args.max_records,
    )

    write_jsonl(output_jsonl, included)

    report = build_report_data(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        records=records,
        included=included,
        skipped=skipped,
        corpus_name=args.corpus_name,
        keep_tooltip=args.keep_tooltip,
        strip_fields=not args.keep_fields,
        allow_interactive_params=args.allow_interactive_params,
        strip_interactions=not args.keep_interactions,
    )
    write_reports(report=report, report_md=report_md, report_json=report_json)

    print(f"Input JSONL: {input_jsonl}")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Input records: {len(records)}")
    print(f"Included records: {len(included)}")
    print(f"Skipped records: {len(skipped)}")
    print(f"Saved Markdown report: {report_md}")
    print(f"Saved JSON report: {report_json}")


if __name__ == "__main__":
    main()
