from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_INPUT_SUFFIXES = {".json", ".jsonl", ".ndjson", ".parquet", ".csv", ".tsv"}
SUPPORTED_CHART_TYPES = {"bar", "line", "area", "point", "circle", "tick", "histogram", "boxplot"}
TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")

CHART_ALIASES = {
    "line": "line",
    "linechart": "line",
    "line_chart": "line",
    "line_plot": "line",
    "timeseries": "line",
    "time_series": "line",
    "step": "line",
    "stem": "line",
    "area": "area",
    "area_chart": "area",
    "fill_between": "area",
    "bar": "bar",
    "barchart": "bar",
    "bar_chart": "bar",
    "column": "bar",
    "column_chart": "bar",
    "broken_barh": "bar",
    "scatter": "point",
    "scatterplot": "point",
    "scatter_plot": "point",
    "scatter_plot_chart": "point",
    "point": "point",
    "point_chart": "point",
    "circle": "circle",
    "circle_chart": "circle",
    "tick": "tick",
    "tick_chart": "tick",
    "hist": "histogram",
    "histogram": "histogram",
    "distribution": "histogram",
    "box": "boxplot",
    "boxplot": "boxplot",
    "box_plot": "boxplot",
    "violin": "boxplot",
    "heatmap": "bar",
    "matshow": "bar",
}

ROLE_BY_CHART_TYPE: dict[str, dict[str, str]] = {
    "bar": {"x": "nominal", "y": "quantitative"},
    "line": {"x": "temporal", "y": "quantitative"},
    "area": {"x": "temporal", "y": "quantitative"},
    "point": {"x": "quantitative", "y": "quantitative"},
    "circle": {"x": "quantitative", "y": "quantitative"},
    "tick": {"x": "quantitative"},
    "histogram": {"x": "quantitative"},
    "boxplot": {"x": "nominal", "y": "quantitative"},
}

AGGREGATE_ALIASES = {
    "avg": "mean",
    "average": "mean",
    "mean": "mean",
    "sum": "sum",
    "total": "sum",
    "count": "count",
    "number": "count",
    "median": "median",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize raw Plot2Code-like files into the prepared ViRAGE VisRAG corpus format."
    )
    parser.add_argument("--src", required=True, help="Path to raw Plot2Code directory or file")
    parser.add_argument(
        "--out",
        default="rag_corpus/data/examples.jsonl",
        help="Output prepared JSONL path. Default: rag_corpus/data/examples.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output manifest path. Default: <out directory>/manifest.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of records to write")
    parser.add_argument("--source-name", default="Plot2Code", help="Source label stored in each corpus row")
    parser.add_argument("--corpus-name", default="plot2code", help="Corpus label stored in each corpus row")
    parser.add_argument("--strict", action="store_true", help="Fail if any raw row cannot be normalized")
    parser.add_argument("--verbose", action="store_true", help="Print normalization diagnostics")
    args = parser.parse_args()

    src_path = Path(args.src).resolve()
    out_path = Path(args.out).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else out_path.parent / "manifest.json"

    if not src_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {src_path}")

    rows = discover_raw_rows(src_path)
    normalized, rejected = normalize_rows(
        rows,
        source_name=args.source_name,
        corpus_name=args.corpus_name,
    )

    if args.strict and rejected:
        details = "; ".join(f"{reason}={count}" for reason, count in sorted(Counter(rejected).items()))
        raise ValueError(f"Failed to normalize {len(rejected)} rows: {details}")

    if args.limit > 0:
        normalized = normalized[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, normalized)
    write_manifest(manifest_path, out_path, normalized, rejected)

    chart_counter = Counter(row["chart_type"] for row in normalized)
    print(f"Source:       {src_path}")
    print(f"Output:       {out_path}")
    print(f"Manifest:     {manifest_path}")
    print(f"Records:      {len(normalized)}")
    print(f"Rejected:     {len(rejected)}")
    if args.verbose and rejected:
        for reason, count in sorted(Counter(rejected).items()):
            print(f"  rejected.{reason}: {count}")
    print("Chart types:")
    for chart_type, count in sorted(chart_counter.items()):
        print(f"  - {chart_type}: {count}")


def discover_raw_rows(src_path: Path) -> list[dict[str, Any]]:
    files = [src_path] if src_path.is_file() else sorted(
        path for path in src_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )
    rows: list[dict[str, Any]] = []
    for path in files:
        for index, row in enumerate(read_file_as_rows(path), start=1):
            item = dict(row)
            item["_source_file"] = path.as_posix()
            item["_source_index"] = index
            rows.append(item)
    return rows


def read_file_as_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            yield payload
        return

    if suffix == ".json":
        yield from unpack_json_payload(json.loads(path.read_text(encoding="utf-8")))
        return

    if suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        yield from df.to_dict(orient="records")
        return

    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        yield from df.to_dict(orient="records")
        return

    if suffix == ".tsv":
        import pandas as pd

        df = pd.read_csv(path, sep="\t")
        yield from df.to_dict(orient="records")
        return

    raise ValueError(f"Unsupported input file type: {path}")


def unpack_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("JSON list must contain objects")
            yield item
        return

    if isinstance(payload, dict):
        for key in ("records", "items", "data", "samples", "examples", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        raise ValueError(f"JSON key {key!r} must contain objects")
                    yield item
                return
        yield payload
        return

    raise ValueError("JSON root must be an object or a list of objects")


def normalize_rows(
        rows: list[dict[str, Any]],
        *,
        source_name: str,
        corpus_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    rejected: list[str] = []
    seen_ids: set[str] = set()

    for row in rows:
        item, reason = normalize_single_row(row, source_name=source_name, corpus_name=corpus_name)
        if item is None:
            rejected.append(reason or "unknown")
            continue
        if item["id"] in seen_ids:
            rejected.append("duplicate_id")
            continue
        seen_ids.add(item["id"])
        normalized.append(item)

    normalized.sort(key=lambda item: (item["chart_type"], item["id"]))
    return normalized, rejected


def normalize_single_row(
        row: dict[str, Any],
        *,
        source_name: str,
        corpus_name: str,
) -> tuple[dict[str, Any] | None, str | None]:
    instruction = first_text(
        row,
        "instruction",
        "prompt",
        "query",
        "question",
        "utterance",
        "text",
        "description",
        "detailed_description",
        "caption",
        "summary",
    )
    description = first_text(row, "description", "detailed_description", "caption", "summary", "text")
    code = first_text(row, "code", "python_code", "source_code", "pyplot_code", "reference_code", "answer")
    source_spec = first_dict(row, "spec_template", "spec", "vega_lite", "vegalite", "vl_spec")
    chart_type = canonicalize_chart_type(
        first_text(row, "chart_type", "plot_type", "chart_family", "mark", "type", "plot_kind")
        or infer_chart_type(instruction=instruction, description=description, code=code,
                            source_file=str(row.get("_source_file", "")), spec=source_spec)
    )

    if not instruction and not description:
        return None, "missing_instruction"
    if chart_type not in SUPPORTED_CHART_TYPES:
        return None, "unsupported_chart_type"

    aggregate = infer_aggregate(" ".join(part for part in [instruction, description, code] if part))
    field_roles = normalize_field_roles(row.get("field_roles") or row.get("encoding_roles"), chart_type)
    spec_template = source_spec if isinstance(source_spec, dict) else build_spec_template(chart_type, field_roles,
                                                                                          aggregate)
    transform_types = infer_transform_types(spec_template=spec_template, aggregate=aggregate, chart_type=chart_type)
    keywords = extract_keywords(row, instruction=instruction, description=description, chart_type=chart_type,
                                aggregate=aggregate)

    example_id = first_text(row, "id", "example_id", "sample_id", "uid", "plot_id") or stable_example_id(row,
                                                                                                         instruction or description or "")
    metadata = {
        "domain": first_text(row, "domain", "topic", "category", "subject") or "general",
        "code_language": infer_code_language(code=code, source_file=str(row.get("_source_file", ""))),
        "source_file": str(row.get("_source_file", "")),
        "source_index": row.get("_source_index"),
    }
    if code:
        metadata["code"] = code

    return {
        "id": example_id,
        "source": source_name,
        "corpus": corpus_name,
        "instruction": instruction or description or "",
        "chart_type": chart_type,
        "description": description,
        "keywords": keywords,
        "field_roles": field_roles,
        "transform_types": transform_types,
        "spec_template": spec_template,
        "metadata": {key: value for key, value in metadata.items() if value is not None},
    }, None


def first_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text
    return None


def first_dict(row: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip().startswith("{"):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
    return None


def stable_example_id(row: dict[str, Any], instruction: str) -> str:
    base = "|".join([
        str(row.get("_source_file", "")),
        str(row.get("_source_index", "")),
        instruction,
    ])
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"plot2code-{digest}"


def canonicalize_chart_type(value: str | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    if normalized in SUPPORTED_CHART_TYPES:
        return normalized
    return CHART_ALIASES.get(normalized, "")


def infer_chart_type(
        *,
        instruction: str | None,
        description: str | None,
        code: str | None,
        source_file: str,
        spec: dict[str, Any] | None,
) -> str:
    if spec:
        mark = spec.get("mark")
        if isinstance(mark, dict):
            mark = mark.get("type")
        chart_type = canonicalize_chart_type(str(mark)) if mark else ""
        if chart_type:
            return chart_type

    text = " ".join(part for part in [instruction, description, code, source_file] if part).lower()
    checks = [
        ("point", ["scatter", "correlation", "relationship", "завис", "связ"]),
        ("histogram", ["histogram", "hist ", "distribution", "density", "распредел"]),
        ("boxplot", ["boxplot", "box plot", "violin"]),
        ("area", ["area chart", "fill_between"]),
        ("bar", ["bar chart", "barplot", "bar ", "column chart", "broken_barh", "compare", "rank", "top"]),
        ("line", ["line chart", "line ", "trend", "time series", "timeseries", "step", "stem", "динами", "тренд"]),
    ]
    for chart_type, hints in checks:
        if any(hint in text for hint in hints):
            return chart_type
    return ""


def infer_aggregate(text: str) -> str | None:
    tokens = {token.lower() for token in TOKEN_RE.findall(text)}
    for token, aggregate in AGGREGATE_ALIASES.items():
        if token in tokens:
            return aggregate
    return None


def normalize_field_roles(value: Any, chart_type: str) -> dict[str, str]:
    if isinstance(value, dict):
        roles = {str(channel): normalize_role(str(role)) for channel, role in value.items() if
                 normalize_role(str(role))}
        if roles:
            return roles
    return dict(ROLE_BY_CHART_TYPE.get(chart_type, {}))


def normalize_role(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {"quantitative", "numeric", "number", "measure", "int", "float", "int64", "float64"}:
        return "quantitative"
    if text in {"temporal", "datetime", "date", "time", "year", "datetime64", "datetime64[ns]"}:
        return "temporal"
    if text in {"nominal", "categorical", "category", "object", "string", "str"}:
        return "nominal"
    if text in {"ordinal", "ordered"}:
        return "ordinal"
    if text in {"bool", "boolean"}:
        return "boolean"
    return ""


def build_spec_template(chart_type: str, field_roles: dict[str, str], aggregate: str | None) -> dict[str, Any]:
    if chart_type == "histogram":
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
        vega_type = "nominal" if role == "boolean" else role
        channel_spec: dict[str, Any] = {"field": f"__{channel}__", "type": vega_type}
        if channel == "y" and aggregate:
            channel_spec["aggregate"] = aggregate
        encoding[channel] = channel_spec

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": chart_type,
        "encoding": encoding,
    }


def infer_transform_types(*, spec_template: dict[str, Any], aggregate: str | None, chart_type: str) -> list[str]:
    transforms: set[str] = set()
    if aggregate:
        transforms.add("aggregate")
    if chart_type == "histogram":
        transforms.add("bin")
    for item in spec_template.get("transform", []) if isinstance(spec_template.get("transform"), list) else []:
        if isinstance(item, dict):
            transforms.update(str(key) for key in item.keys())
    for channel_spec in (spec_template.get("encoding") or {}).values():
        if isinstance(channel_spec, dict):
            if channel_spec.get("aggregate"):
                transforms.add("aggregate")
            if channel_spec.get("bin"):
                transforms.add("bin")
            if channel_spec.get("timeUnit"):
                transforms.add("timeUnit")
    return sorted(transforms)


def extract_keywords(
        row: dict[str, Any],
        *,
        instruction: str | None,
        description: str | None,
        chart_type: str,
        aggregate: str | None,
) -> list[str]:
    explicit = row.get("keywords") or row.get("tags")
    keywords: list[str] = []
    if isinstance(explicit, list):
        keywords.extend(str(item).strip().lower() for item in explicit if str(item).strip())
    elif isinstance(explicit, str) and explicit.strip():
        keywords.extend(part.strip().lower() for part in re.split(r"[;,|]", explicit) if part.strip())

    text = " ".join(part for part in [instruction, description] if part).lower()
    keywords.append(chart_type)
    if aggregate:
        keywords.append(aggregate)
    if any(token in text for token in ["trend", "time", "timeline", "date", "series", "динам", "тренд"]):
        keywords.append("trend")
    if any(token in text for token in ["compare", "comparison", "rank", "top", "bottom", "сравн"]):
        keywords.append("comparison")
    if any(token in text for token in ["distribution", "spread", "density", "распредел"]):
        keywords.append("distribution")
    if any(token in text for token in ["relationship", "correlation", "scatter", "связ", "завис"]):
        keywords.append("relationship")
    return dedupe(keywords)


def dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def infer_code_language(*, code: str | None, source_file: str) -> str | None:
    text = f"{source_file}\n{code or ''}".lower()
    if source_file.lower().endswith(".r") or "ggplot" in text or "<-" in text:
        return "r"
    if source_file.lower().endswith(".py") or "matplotlib" in text or "plotly" in text or "plt." in text:
        return "python"
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_manifest(path: Path, corpus_path: Path, rows: list[dict[str, Any]], rejected: list[str]) -> None:
    chart_counts = Counter(row["chart_type"] for row in rows)
    payload = {
        "version": "1",
        "format": "virage-visrag-prepared-corpus-v1",
        "files": [corpus_path.name],
        "records": len(rows),
        "rejected_records": len(rejected),
        "chart_type_counts": dict(sorted(chart_counts.items())),
        "rejection_counts": dict(sorted(Counter(rejected).items())),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
