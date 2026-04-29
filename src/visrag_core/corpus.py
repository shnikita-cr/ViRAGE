from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")
_SUPPORTED_CHARTS = {"line", "bar", "point", "histogram", "boxplot", "area", "circle", "tick"}
_CHART_ALIASES = {
    "line": "line",
    "linechart": "line",
    "line_chart": "line",
    "timeseries": "line",
    "time_series": "line",
    "trend": "line",
    "area": "area",
    "area_chart": "area",
    "bar": "bar",
    "barchart": "bar",
    "bar_chart": "bar",
    "column": "bar",
    "column_chart": "bar",
    "grouped_bar": "bar",
    "stacked_bar": "bar",
    "scatter": "point",
    "scatterplot": "point",
    "scatter_plot": "point",
    "scatter chart": "point",
    "scatter plot": "point",
    "point": "point",
    "circle": "circle",
    "hist": "histogram",
    "histogram": "histogram",
    "distribution": "histogram",
    "box": "boxplot",
    "boxplot": "boxplot",
    "box_plot": "boxplot",
    "tick": "tick",
    "heatmap": "bar",
}


@dataclass(slots=True)
class CorpusSpec:
    name: str
    role: str
    filenames: tuple[str, ...]
    weight: float = 1.0


@dataclass(slots=True)
class CorpusRecord:
    example_id: str
    source: str
    corpus: str
    chart_type: str
    instruction: str
    description: str | None
    tags: list[str] = field(default_factory=list)
    code_language: str | None = None
    domain: str | None = None
    spec_template: dict[str, Any] | None = None
    field_roles: dict[str, str] = field(default_factory=dict)
    transform_types: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_text(self) -> str:
        parts = [
            f"corpus: {self.corpus}",
            f"source: {self.source}",
            f"chart_type: {self.chart_type}",
            f"instruction: {self.instruction}",
        ]
        if self.description:
            parts.append(f"description: {self.description}")
        if self.tags:
            parts.append(f"tags: {', '.join(self.tags)}")
        if self.field_roles:
            parts.append(f"field_roles: {json.dumps(self.field_roles, ensure_ascii=False, sort_keys=True)}")
        if self.transform_types:
            parts.append(f"transforms: {', '.join(self.transform_types)}")
        if self.domain:
            parts.append(f"domain: {self.domain}")
        if self.code_language:
            parts.append(f"code_language: {self.code_language}")
        return "\n".join(parts)


DEFAULT_CORPUS_REGISTRY: tuple[CorpusSpec, ...] = (
    CorpusSpec(name="plot2code", role="reference", filenames=("plot2code.jsonl", "plot2code.json"), weight=1.0),
    CorpusSpec(name="chartmimic", role="high_quality_selection", filenames=("chartmimic.jsonl", "chartmimic.json"),
               weight=1.2),
    CorpusSpec(name="chart2code_160k", role="implementation",
               filenames=("chart2code_160k.jsonl", "chart2code_160k.json"), weight=0.9),
    CorpusSpec(name="chartx", role="multimodal_support", filenames=("chartx.jsonl", "chartx.json"), weight=1.05),
)


def canonicalize_chart_type(value: str | None) -> str:
    if not value:
        return ""
    text = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    if normalized in _SUPPORTED_CHARTS:
        return normalized
    return _CHART_ALIASES.get(normalized) or _CHART_ALIASES.get(text, "")


def tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token.lower() for token in _TOKEN_RE.findall(text) if token.strip()}


def resolve_available_corpora(corpus_root: Path, registry: Iterable[CorpusSpec] = DEFAULT_CORPUS_REGISTRY) -> list[
    tuple[CorpusSpec, Path]]:
    result: list[tuple[CorpusSpec, Path]] = []
    for spec in registry:
        for filename in spec.filenames:
            candidates = [corpus_root / filename, corpus_root / spec.name / filename]
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    result.append((spec, candidate))
                    break
            else:
                continue
            break
    return result


def load_normalized_records(path: Path, corpus_name: str) -> list[CorpusRecord]:
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".jsonl":
            for raw in path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if raw:
                    payload = json.loads(raw)
                    if isinstance(payload, dict):
                        rows.append(payload)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows = [row for row in payload if isinstance(row, dict)]
            elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
                rows = [row for row in payload["records"] if isinstance(row, dict)]
    except Exception:
        return []

    records: list[CorpusRecord] = []
    for idx, row in enumerate(rows, start=1):
        chart_type = canonicalize_chart_type(_first_text(row, "chart_type", "chart_family", "mark", "type"))
        spec_template = _extract_spec_template(row)
        if not chart_type and spec_template:
            chart_type = _mark_from_spec(spec_template)
        instruction = _first_text(row, "instruction", "prompt", "query", "text", "utterance")
        if not chart_type or not instruction:
            continue
        example_id = _first_text(row, "id", "example_id", "sample_id") or f"{corpus_name}-{idx:05d}"
        field_roles = _coerce_string_map(row.get("field_roles") or row.get("encoding_roles"))
        if not field_roles and spec_template:
            field_roles = _field_roles_from_spec(spec_template)
        transform_types = _coerce_string_list(row.get("transform_types") or row.get("transforms"))
        if not transform_types and spec_template:
            transform_types = _transform_types_from_spec(spec_template)
        records.append(
            CorpusRecord(
                example_id=example_id,
                source=_first_text(row, "source") or corpus_name,
                corpus=corpus_name,
                chart_type=chart_type,
                instruction=instruction,
                description=_first_text(row, "description", "summary", "caption"),
                tags=_coerce_tags(row.get("tags")),
                code_language=_first_text(row, "code_language", "language"),
                domain=_first_text(row, "domain", "topic", "category"),
                spec_template=spec_template,
                field_roles=field_roles,
                transform_types=transform_types,
                metadata={
                    "source_file": row.get("source_file"),
                    "quality_tier": row.get("quality_tier"),
                    "complexity": row.get("complexity"),
                },
            )
        )
    return records


def _extract_spec_template(row: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("spec_template", "spec", "vega_lite", "vegalite", "vl_spec"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _mark_from_spec(spec: dict[str, Any]) -> str:
    mark = spec.get("mark")
    if isinstance(mark, dict):
        return canonicalize_chart_type(str(mark.get("type") or ""))
    if isinstance(mark, str):
        return canonicalize_chart_type(mark)
    return ""


def _field_roles_from_spec(spec: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    encoding = spec.get("encoding")
    if not isinstance(encoding, dict):
        return result
    for channel, channel_spec in encoding.items():
        if isinstance(channel_spec, dict):
            field_type = channel_spec.get("type")
            if isinstance(field_type, str):
                result[str(channel)] = field_type
    return result


def _transform_types_from_spec(spec: dict[str, Any]) -> list[str]:
    transforms = spec.get("transform")
    if not isinstance(transforms, list):
        return []
    result: list[str] = []
    for transform in transforms:
        if not isinstance(transform, dict):
            continue
        for key in (
        "aggregate", "joinaggregate", "filter", "calculate", "bin", "timeUnit", "window", "density", "loess",
        "regression"):
            if key in transform and key not in result:
                result.append(key)
    return result


def _first_text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _coerce_tags(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in re.split(r"[;,|]", value) if part.strip()]
    return []


def _coerce_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _coerce_string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if str(k).strip() and str(v).strip()}
