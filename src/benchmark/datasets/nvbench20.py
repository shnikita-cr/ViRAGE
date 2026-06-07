from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.benchmark.datasets.nvbench20_io import (
    chart_type as _chart_type,
    looks_like_date as _looks_like_date,
    relative_path as _relative_path,
    require_dir as _require_dir,
    require_file as _require_file,
    skip_record as _skip,
    split_table_file_name as _split_table_file_name,
    string_list as _string_list,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
    write_markdown_report as _write_markdown_report,
)


class NVBench20ConversionError(RuntimeError):
    pass


def convert_nvbench20(
    *,
    input_path: Path,
    database_csv_dir: Path,
    output_dir: Path,
    limit: int | None,
    seed: int,
    single_table_only: bool,
) -> dict[str, Any]:
    _require_file(input_path, "nvBench 2.0 parquet")
    _require_dir(database_csv_dir, "nvBench 2.0 database_csv directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = output_dir / "reference_specs"
    reference_dir.mkdir(parents=True, exist_ok=True)
    csv_index = _build_csv_index(database_csv_dir)
    frame = pd.read_parquet(input_path)
    rows = _frame_records(frame, seed=seed, shuffle=limit is not None)
    cases, skipped, processed_rows = _convert_rows(
        rows=rows,
        csv_index=csv_index,
        output_dir=output_dir,
        reference_dir=reference_dir,
        single_table_only=single_table_only,
        export_limit=limit,
    )
    _write_jsonl(output_dir / "cases.jsonl", cases)
    _write_jsonl(output_dir / "skipped_cases.jsonl", skipped)
    report = _conversion_report(
        input_path=input_path,
        database_csv_dir=database_csv_dir,
        output_dir=output_dir,
        total_rows=len(frame),
        selected_rows=processed_rows,
        cases=cases,
        skipped=skipped,
        seed=seed,
        limit=limit,
        single_table_only=single_table_only,
    )
    _write_json(output_dir / "conversion_report.json", report)
    _write_markdown_report(output_dir / "conversion_report.md", report)
    return report


def _convert_rows(
    *,
    rows: list[tuple[int, dict[str, Any]]],
    csv_index: dict[tuple[str, ...], list[Path]],
    output_dir: Path,
    reference_dir: Path,
    single_table_only: bool,
    export_limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    cases: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    processed_rows = 0
    for source_index, row in rows:
        if export_limit is not None and len(cases) >= export_limit:
            break
        processed_rows += 1
        exported, skip = _convert_row(
            row=row,
            source_index=source_index,
            csv_index=csv_index,
            output_dir=output_dir,
            reference_dir=reference_dir,
            single_table_only=single_table_only,
        )
        if skip is not None:
            skipped.append(skip)
        elif exported is not None:
            cases.append(exported)
    return cases, skipped, processed_rows


def _convert_row(
    *,
    row: dict[str, Any],
    source_index: int,
    csv_index: dict[tuple[str, ...], list[Path]],
    output_dir: Path,
    reference_dir: Path,
    single_table_only: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    query = str(row.get("nl_query") or "").strip()
    if not query:
        return None, _skip(source_index, "empty_query")
    schema = _loads_object(row.get("table_schema"), label="table_schema", source_index=source_index)
    gold_answer = _loads_list(row.get("gold_answer"), label="gold_answer", source_index=source_index)
    if not isinstance(schema, dict):
        return None, _skip(source_index, "invalid_table_schema")
    table_columns = _string_list(schema.get("table_columns"))
    if not table_columns:
        return None, _skip(source_index, "empty_table_columns")
    candidates = csv_index.get(tuple(table_columns)) or []
    if single_table_only and len(candidates) != 1:
        reason = "missing_csv" if not candidates else "ambiguous_csv_match"
        return None, _skip(source_index, reason, table_columns=table_columns, candidate_count=len(candidates))
    if not candidates:
        return None, _skip(source_index, "missing_csv", table_columns=table_columns)
    reference_specs = _reference_specs(gold_answer=gold_answer, schema=schema)
    if not reference_specs:
        return None, _skip(source_index, "invalid_reference_specs")
    return _case_payload(
        source_index=source_index,
        query=query,
        csv_path=candidates[0],
        reference_specs=reference_specs,
        output_dir=output_dir,
        reference_dir=reference_dir,
    ), None


def _case_payload(
    *,
    source_index: int,
    query: str,
    csv_path: Path,
    reference_specs: list[dict[str, Any]],
    output_dir: Path,
    reference_dir: Path,
) -> dict[str, Any]:
    database_id, table_name = _split_table_file_name(csv_path)
    case_id = f"nvbench20_{source_index:06d}"
    reference_path = reference_dir / f"{case_id}.json"
    _write_json(reference_path, reference_specs)
    return {
        "case_id": case_id,
        "dataset_name": "nvbench20",
        "query": query,
        "data_path": _relative_path(csv_path, output_dir),
        "reference_spec": reference_specs[0],
        "reference_specs": reference_specs,
        "metadata": {
            "source_row_index": source_index,
            "database_id": database_id,
            "table_name": table_name,
            "reference_count": len(reference_specs),
            "chart_type": _chart_type(reference_specs[0]),
            "benchmark_mode": "single_table",
            "steps_hidden": True,
            "reference_specs_path": _relative_path(reference_path, output_dir),
        },
    }


def _reference_specs(*, gold_answer: list[Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
    field_types = _field_types(schema)
    specs = [_normalize_reference_spec(item, field_types=field_types) for item in gold_answer if isinstance(item, dict)]
    return [item for item in specs if item]


def _build_csv_index(database_csv_dir: Path) -> dict[tuple[str, ...], list[Path]]:
    index: dict[tuple[str, ...], list[Path]] = {}
    for path in sorted(database_csv_dir.glob("*.csv")):
        columns = _csv_columns(path)
        if columns:
            index.setdefault(tuple(columns), []).append(path)
    if not index:
        raise NVBench20ConversionError(f"No readable CSV tables found in {database_csv_dir}")
    return index


def _csv_columns(path: Path) -> list[str]:
    try:
        return [str(column) for column in pd.read_csv(path, nrows=0).columns]
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError):
        return []


def _normalize_reference_spec(spec: dict[str, Any], *, field_types: dict[str, str]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(spec))
    normalized.setdefault("$schema", "https://vega.github.io/schema/vega-lite/v5.json")
    normalized["mark"] = _normalize_mark(normalized.get("mark"))
    encoding = normalized.get("encoding")
    if isinstance(encoding, dict):
        normalized["encoding"] = {
            channel: _normalize_encoding(value, field_types=field_types)
            for channel, value in encoding.items()
            if isinstance(value, dict)
        }
    return normalized


def _normalize_mark(mark: Any) -> Any:
    if isinstance(mark, str):
        return "arc" if mark == "pie" else mark
    if isinstance(mark, dict):
        clone = dict(mark)
        if clone.get("type") == "pie":
            clone["type"] = "arc"
        return clone
    return mark


def _normalize_encoding(value: dict[str, Any], *, field_types: dict[str, str]) -> dict[str, Any]:
    clone = dict(value)
    aggregate = clone.get("aggregate")
    field = clone.get("field")
    if "type" not in clone:
        if isinstance(aggregate, str) and aggregate.strip():
            clone["type"] = "quantitative"
        elif isinstance(field, str) and field in field_types:
            clone["type"] = field_types[field]
    return clone


def _field_types(schema: dict[str, Any]) -> dict[str, str]:
    examples = schema.get("column_examples") or {}
    if not isinstance(examples, dict):
        return {}
    return {str(field): _infer_type(values) for field, values in examples.items()}


def _infer_type(values: Any) -> str:
    if not isinstance(values, list):
        return "nominal"
    cleaned = [item for item in values if item is not None]
    if cleaned and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in cleaned):
        return "quantitative"
    if cleaned and all(_looks_like_date(str(item)) for item in cleaned):
        return "temporal"
    return "nominal"


def _loads_object(raw: Any, *, label: str, source_index: int) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NVBench20ConversionError(f"Invalid {label} JSON at row {source_index}: {exc}") from exc
    return parsed if isinstance(parsed, dict) else None


def _loads_list(raw: Any, *, label: str, source_index: int) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NVBench20ConversionError(f"Invalid {label} JSON at row {source_index}: {exc}") from exc
    return parsed if isinstance(parsed, list) else []


def _frame_records(frame: pd.DataFrame, *, seed: int, shuffle: bool) -> list[tuple[int, dict[str, Any]]]:
    source = frame.sample(frac=1.0, random_state=seed) if shuffle else frame
    return [(int(index), dict(row)) for index, row in source.iterrows()]


def _conversion_report(**values: Any) -> dict[str, Any]:
    cases = values["cases"]
    skipped = values["skipped"]
    chart_types = Counter(str(item["metadata"].get("chart_type") or "unknown") for item in cases)
    reference_counts = Counter(int(item["metadata"].get("reference_count") or 0) for item in cases)
    return {
        "input": str(values["input_path"]),
        "database_csv_dir": str(values["database_csv_dir"]),
        "output_dir": str(values["output_dir"]),
        "total_rows": values["total_rows"],
        "selected_rows": values["selected_rows"],
        "exported_single_table_cases": len(cases),
        "skipped_cases": len(skipped),
        "skip_reasons": dict(sorted(Counter(str(item.get("skip_reason")) for item in skipped).items())),
        "chart_type_distribution": dict(sorted(chart_types.items())),
        "reference_count_distribution": {str(key): value for key, value in sorted(reference_counts.items())},
        "seed": values["seed"],
        "limit": values["limit"],
        "single_table_only": values["single_table_only"],
    }
