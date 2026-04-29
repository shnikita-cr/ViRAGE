from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_MARK_TYPES = {
    "area", "bar", "circle", "line", "point", "rect", "rule", "square", "text", "tick",
    "boxplot", "errorbar", "errorband", "geoshape",
}
SUPPORTED_CHART_PATTERNS = {
    "area_chart", "bar_chart", "boxplot", "choropleth_map", "errorbar", "errorband",
    "faceted_chart", "grouped_bar", "heatmap", "histogram", "interactive_chart", "layered_chart",
    "line_chart", "multi_series_line", "scatter_plot", "stacked_bar", "streamgraph", "table",
    "unknown",
}
SUPPORTED_CORPUS_TYPES = {
    "ambiguity_case", "dataset_profile", "design_constraint", "example", "failure_case",
    "rule", "success_case", "utterance",
}
SUPPORTED_ROLES = {"quantitative", "temporal", "ordinal", "nominal", "geojson", "boolean", "unknown"}
SUPPORTED_CHANNELS = {
    "x", "y", "color", "size", "shape", "opacity", "row", "column", "theta", "radius",
    "detail", "tooltip", "longitude", "latitude", "text",
}
CONTROL_FILES = {
    "manifest.json", "validation_report.json", "lexical_index.json", "corpus_manifest.json",
    "corpus_coverage.json", "corpus_validation_errors.json", "autorag_export_report.json",
}
JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}
TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")

MARK_ALIASES = {
    "bar_chart": "bar", "barchart": "bar", "column": "bar", "column_chart": "bar",
    "line_chart": "line", "linechart": "line", "time_series": "line", "timeseries": "line",
    "scatter": "point", "scatter_plot": "point", "scatterplot": "point",
    "heatmap": "rect", "heat_map": "rect", "imshow": "rect", "matshow": "rect",
    "hist": "bar", "histogram": "bar",
    "box": "boxplot", "box_plot": "boxplot",
}

ROLE_ALIASES = {
    "bool": "boolean", "boolean": "boolean",
    "category": "nominal", "categorical": "nominal", "dimension": "nominal", "object": "nominal",
    "str": "nominal", "string": "nominal",
    "date": "temporal", "datetime": "temporal", "datetime64": "temporal", "datetime64[ns]": "temporal",
    "time": "temporal", "timestamp": "temporal", "year": "temporal",
    "float": "quantitative", "float64": "quantitative", "int": "quantitative", "int64": "quantitative",
    "measure": "quantitative", "number": "quantitative", "numeric": "quantitative", "quantitative": "quantitative",
    "ordered": "ordinal", "ordinal": "ordinal",
    "geojson": "geojson",
}


def iter_json_records(root: Path, *, include_control_files: bool = False) -> Iterable[dict[str, Any]]:
    paths = [root] if root.is_file() else sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in JSON_SUFFIXES
        and (include_control_files or path.name not in CONTROL_FILES)
    )
    for path in paths:
        yield from read_json_records(path)


def read_json_records(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    decoder = json.JSONDecoder()
    index = 0
    yielded = False
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            payload, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON record stream in {path}: {exc}") from exc
        yield from unpack_json_payload(payload, path)
        yielded = True
        index = end
    if not yielded:
        return


def unpack_json_payload(payload: Any, path: Path) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"JSON array in {path} must contain objects.")
            yield item
        return
    if isinstance(payload, dict):
        for key in ("records", "items", "data", "examples", "rows", "qa"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        raise ValueError(f"JSON key {key!r} in {path} must contain objects.")
                    yield item
                return
        yield payload
        return
    raise ValueError(f"Unsupported JSON payload in {path}; expected object or array.")


def normalize_record(row: dict[str, Any], *, default_corpus: str) -> dict[str, Any]:
    record = dict(row)
    record_id = str(record.get("id") or record.get("doc_id") or record.get("example_id") or "").strip()
    instruction = _first_text(record, "instruction", "query", "utterance", "question", "description", "summary")
    corpus_type = str(record.get("corpus_type") or infer_corpus_type(default_corpus)).strip() or "example"
    source = str(record.get("source") or default_corpus).strip()
    mark_type = canonical_mark_type(record.get("mark_type") or record.get("chart_type") or _mark_from_spec(record))
    chart_pattern = normalize_chart_pattern(record.get("chart_pattern") or infer_chart_pattern(mark_type, record))
    task_type = str(record.get("task_type") or infer_task_type(chart_pattern)).strip() or "unknown"
    field_roles = normalize_field_roles(
        record.get("field_roles")
        or record.get("expected_field_roles")
        or _field_roles_from_possible_interpretation(record)
        or {},
    )
    spec_template = record.get("spec_template") or record.get("spec") or build_minimal_spec_template(mark_type, field_roles, chart_pattern)
    if not isinstance(spec_template, dict):
        spec_template = None
    if not record_id:
        record_id = stable_record_id(source, corpus_type, instruction or json.dumps(record, sort_keys=True, default=str))
    normalized = {
        "schema_version": str(record.get("schema_version") or "1.0"),
        "id": record_id,
        "source": source,
        "source_url": record.get("source_url"),
        "license": record.get("license"),
        "corpus_type": corpus_type,
        "quality_tier": str(record.get("quality_tier") or record.get("quality") or default_quality(corpus_type)),
        "source_weight": float(record.get("source_weight") or default_source_weight(corpus_type)),
        "language": str(record.get("language") or "en"),
        "domain": str(record.get("domain") or "generic"),
        "instruction": instruction or build_instruction_from_record(record, corpus_type),
        "query_variants": normalize_text_list(record.get("query_variants") or record.get("queries") or []),
        "mark_type": mark_type,
        "chart_type": mark_type,
        "chart_pattern": chart_pattern,
        "task_type": task_type,
        "field_roles": field_roles,
        "required_channels": normalize_text_list(record.get("required_channels") or list(field_roles.keys())),
        "optional_channels": normalize_text_list(record.get("optional_channels") or []),
        "transforms": normalize_text_list(record.get("transforms") or record.get("transform_types") or []),
        "transform_types": normalize_text_list(record.get("transform_types") or record.get("transforms") or []),
        "data_constraints": normalize_text_list(record.get("data_constraints") or []),
        "negative_conditions": normalize_text_list(record.get("negative_conditions") or []),
        "spec_template": spec_template,
        "validation": record.get("validation") if isinstance(record.get("validation"), dict) else {},
        "tags": normalize_text_list(record.get("tags") or record.get("keywords") or []),
        "keywords": normalize_text_list(record.get("keywords") or record.get("tags") or [chart_pattern, task_type]),
        "metadata": normalize_metadata(record),
    }
    return normalized


def validate_record(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not str(row.get("id") or "").strip():
        errors.append("missing_id")
    corpus_type = str(row.get("corpus_type") or "").strip()
    if corpus_type not in SUPPORTED_CORPUS_TYPES:
        errors.append(f"unsupported_corpus_type:{corpus_type}")
    instruction = str(row.get("instruction") or "").strip()
    if not instruction:
        errors.append("missing_instruction")
    mark_type = str(row.get("mark_type") or row.get("chart_type") or "").strip()
    if mark_type and mark_type not in SUPPORTED_MARK_TYPES:
        errors.append(f"unsupported_mark_type:{mark_type}")
    chart_pattern = str(row.get("chart_pattern") or "").strip()
    if chart_pattern and chart_pattern not in SUPPORTED_CHART_PATTERNS:
        # Unknown patterns are allowed for future corpora, but report them as warning-like validation error.
        errors.append(f"unsupported_chart_pattern:{chart_pattern}")
    field_roles = row.get("field_roles") or {}
    if not isinstance(field_roles, dict):
        errors.append("field_roles_not_object")
    else:
        for channel, role in field_roles.items():
            if str(channel) not in SUPPORTED_CHANNELS:
                errors.append(f"unsupported_channel:{channel}")
            if str(role) not in SUPPORTED_ROLES:
                errors.append(f"unsupported_role:{role}")
    spec_template = row.get("spec_template")
    if spec_template is not None and not isinstance(spec_template, dict):
        errors.append("spec_template_not_object_or_null")
    return errors


def normalize_qa_record(row: dict[str, Any]) -> dict[str, Any]:
    qid = str(row.get("qid") or row.get("id") or "").strip()
    query = str(row.get("query") or row.get("question") or "").strip()
    if not qid or not query:
        raise ValueError(f"QA record must contain qid/id and query/question: {row!r}")
    retrieval_gt = normalize_retrieval_gt(row.get("retrieval_gt"))
    generation_gt = row.get("generation_gt") or row.get("answer") or row.get("expected_answer") or ""
    return {
        "qid": qid,
        "query": query,
        "retrieval_gt": retrieval_gt,
        "generation_gt": generation_gt,
        "group": row.get("group") or row.get("benchmark_group") or "benchmark_01_autorag_retrieval_main",
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
    }


def normalize_retrieval_gt(value: Any) -> list[list[str]]:
    if isinstance(value, str):
        return [[value]]
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return [[str(item) for item in value]]
        if all(isinstance(item, list) for item in value):
            return [[str(inner) for inner in group] for group in value]
    raise ValueError(f"retrieval_gt must be string, list[str], or list[list[str]], got: {value!r}")


def flatten_retrieval_gt(value: list[list[str]]) -> list[str]:
    result: list[str] = []
    for group in value:
        for item in group:
            if item not in result:
                result.append(item)
    return result


def autorag_contents(row: dict[str, Any]) -> str:
    parts = [
        ("title", row.get("instruction")),
        ("corpus_type", row.get("corpus_type")),
        ("source", row.get("source")),
        ("quality_tier", row.get("quality_tier")),
        ("mark_type", row.get("mark_type")),
        ("chart_pattern", row.get("chart_pattern")),
        ("task_type", row.get("task_type")),
        ("field_roles", json.dumps(row.get("field_roles") or {}, ensure_ascii=False)),
        ("required_channels", ", ".join(row.get("required_channels") or [])),
        ("transforms", ", ".join(row.get("transforms") or row.get("transform_types") or [])),
        ("data_constraints", ", ".join(row.get("data_constraints") or [])),
        ("negative_conditions", ", ".join(row.get("negative_conditions") or [])),
        ("query_variants", ", ".join(row.get("query_variants") or [])),
    ]
    return "\n".join(f"{key}: {value}" for key, value in parts if value not in (None, "", [], {}))


def autorag_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "last_modified_datetime": datetime.now(timezone.utc).replace(tzinfo=None),
        "source": row.get("source"),
        "source_url": row.get("source_url"),
        "corpus_type": row.get("corpus_type"),
        "quality_tier": row.get("quality_tier"),
        "source_weight": row.get("source_weight"),
        "mark_type": row.get("mark_type"),
        "chart_pattern": row.get("chart_pattern"),
        "task_type": row.get("task_type"),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def stable_record_id(source: str, corpus_type: str, content: str) -> str:
    digest = hashlib.sha1(f"{source}|{corpus_type}|{content}".encode("utf-8")).hexdigest()[:12]
    return f"{source}_{corpus_type}_{digest}".replace(" ", "_").lower()


def canonical_mark_type(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("type")
    text = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower()).strip("_")
    if text in SUPPORTED_MARK_TYPES:
        return text
    return MARK_ALIASES.get(text, "")


def normalize_chart_pattern(value: Any) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    aliases = {
        "aggregate_bar": "bar_chart",
        "bar": "bar_chart",
        "line": "line_chart",
        "scatter": "scatter_plot",
        "point": "scatter_plot",
        "area": "area_chart",
        "rect": "heatmap",
    }
    text = aliases.get(text, text)
    return text if text in SUPPORTED_CHART_PATTERNS else "unknown"


def infer_chart_pattern(mark_type: str, row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(key) or "") for key in ("instruction", "description", "task_type")).lower()
    if "hist" in text or "distribution" in text:
        return "histogram"
    if "scatter" in text or "correlation" in text or "relationship" in text:
        return "scatter_plot"
    if "heat" in text:
        return "heatmap"
    if mark_type == "line":
        return "line_chart"
    if mark_type == "bar":
        return "bar_chart"
    if mark_type == "area":
        return "area_chart"
    if mark_type == "boxplot":
        return "boxplot"
    if mark_type == "rect":
        return "heatmap"
    return "unknown"


def infer_task_type(chart_pattern: str) -> str:
    if chart_pattern in {"line_chart", "area_chart", "multi_series_line", "streamgraph"}:
        return "trend"
    if chart_pattern in {"histogram", "boxplot"}:
        return "distribution"
    if chart_pattern == "scatter_plot":
        return "relationship"
    if chart_pattern in {"bar_chart", "stacked_bar", "grouped_bar"}:
        return "comparison"
    if chart_pattern == "heatmap":
        return "matrix"
    return "unknown"


def normalize_field_roles(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for channel, role in value.items():
        channel_name = str(channel)
        role_name = normalize_role(str(role))
        if channel_name in SUPPORTED_CHANNELS and role_name:
            result[channel_name] = role_name
    return result


def normalize_role(value: str) -> str:
    text = value.strip().lower()
    return ROLE_ALIASES.get(text, text if text in SUPPORTED_ROLES else "")


def normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str) and value.strip():
        values = [part.strip() for part in re.split(r"[;,|]", value) if part.strip()]
    else:
        values = []
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def normalize_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    ignored = {
        "schema_version", "id", "source", "source_url", "license", "corpus_type", "quality_tier",
        "source_weight", "language", "domain", "instruction", "query_variants", "mark_type", "chart_type",
        "chart_pattern", "task_type", "field_roles", "required_channels", "optional_channels", "transforms",
        "transform_types", "data_constraints", "negative_conditions", "spec_template", "validation", "tags",
        "keywords", "metadata",
    }
    extra = {key: value for key, value in row.items() if key not in ignored}
    return {**metadata, **extra}


def build_instruction_from_record(row: dict[str, Any], corpus_type: str) -> str:
    if corpus_type == "dataset_profile":
        dataset_id = row.get("dataset_id") or row.get("id") or "dataset"
        columns = row.get("columns") if isinstance(row.get("columns"), list) else []
        column_names = [str(col.get("name")) for col in columns if isinstance(col, dict) and col.get("name")]
        return f"Dataset profile for {dataset_id}: {', '.join(column_names)}"
    return str(row.get("id") or corpus_type)


def build_minimal_spec_template(mark_type: str, field_roles: dict[str, str], chart_pattern: str) -> dict[str, Any] | None:
    if not mark_type or not field_roles:
        return None
    if chart_pattern == "histogram":
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": "bar",
            "encoding": {
                "x": {"field": "__x__", "type": "quantitative", "bin": True},
                "y": {"aggregate": "count", "type": "quantitative"},
            },
        }
    encoding: dict[str, dict[str, Any]] = {}
    for channel, role in field_roles.items():
        if channel in {"detail", "tooltip"}:
            continue
        encoding[channel] = {"field": f"__{channel}__", "type": "nominal" if role == "boolean" else role}
    return {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "mark": mark_type, "encoding": encoding}


def _first_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _mark_from_spec(row: dict[str, Any]) -> str:
    spec = row.get("spec_template") or row.get("spec")
    if not isinstance(spec, dict):
        return ""
    mark = spec.get("mark")
    if isinstance(mark, dict):
        return str(mark.get("type") or "")
    return str(mark or "")


def _field_roles_from_possible_interpretation(row: dict[str, Any]) -> dict[str, Any]:
    interpretations = row.get("possible_interpretations")
    if isinstance(interpretations, list) and interpretations:
        first = interpretations[0]
        if isinstance(first, dict):
            return first.get("field_roles") or first.get("expected_field_roles") or {}
    return {}


def default_quality(corpus_type: str) -> str:
    return "gold" if corpus_type in {"example", "rule", "design_constraint", "dataset_profile"} else "silver"


def default_source_weight(corpus_type: str) -> float:
    return {
        "example": 1.0,
        "rule": 0.9,
        "design_constraint": 0.9,
        "dataset_profile": 0.7,
        "success_case": 0.8,
        "failure_case": 0.7,
        "utterance": 0.6,
        "ambiguity_case": 0.6,
    }.get(corpus_type, 0.5)


def infer_corpus_type(default_corpus: str) -> str:
    key = default_corpus.lower()
    if "rule" in key or "docs" in key:
        return "rule"
    if "constraint" in key or "draco" in key or "compass" in key:
        return "design_constraint"
    if "dataset" in key or "profile" in key:
        return "dataset_profile"
    if "failure" in key:
        return "failure_case"
    if "success" in key:
        return "success_case"
    if "utterance" in key or "nlv" in key:
        return "utterance"
    if "ambigu" in key:
        return "ambiguity_case"
    return "example"


def coverage(rows: list[dict[str, Any]], qa_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    counters = {
        "corpus_type": Counter(str(row.get("corpus_type") or "") for row in rows),
        "mark_type": Counter(str(row.get("mark_type") or "") for row in rows),
        "chart_pattern": Counter(str(row.get("chart_pattern") or "") for row in rows),
        "task_type": Counter(str(row.get("task_type") or "") for row in rows),
        "quality_tier": Counter(str(row.get("quality_tier") or "") for row in rows),
    }
    result: dict[str, Any] = {key: dict(counter) for key, counter in counters.items()}
    result["records"] = len(rows)
    if qa_rows is not None:
        result["qa_records"] = len(qa_rows)
        group_counter = Counter(str(row.get("group") or "") for row in qa_rows)
        result["qa_groups"] = dict(group_counter)
    return result
