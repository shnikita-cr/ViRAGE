from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_NAME = "global_analysis.csv"

NUMBERED_PREFIX_RE = re.compile(r"^\d{3,6}_")
RUN_TIMESTAMP_RE = re.compile(
    r"(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}[-:]\d{2}[-:]\d{2})"
)


def strip_number_prefix(value: str) -> str:
    return NUMBERED_PREFIX_RE.sub("", value)


def normalize_stem(path: Path) -> str:
    return strip_number_prefix(path.stem)


def is_logical_artifact_name(stem: str, logical_name: str) -> bool:
    if stem == logical_name:
        return True
    # Current node artifacts may include semantic/technical attempt suffixes, for example:
    # 012_vega_spec_semantic_001_technical_002.json. Treat these as the same logical node
    # while preserving separate files on disk.
    return (
        stem.startswith(f"{logical_name}_semantic_")
        or stem.startswith(f"{logical_name}_technical_")
        or stem.startswith(f"{logical_name}_attempt_")
    )


def read_text_safe(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def read_json_safe(path: Path) -> Any | None:
    text = read_text_safe(path)

    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        return None


def dump_json_cell(value: Any) -> str:
    if value is None:
        return ""

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def ensure_aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def parse_datetime_safe(value: Any) -> datetime | None:
    if not value:
        return None

    if isinstance(value, datetime):
        return ensure_aware_utc(value)

    text = str(value).strip()

    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    # Run folder names often use 2026-05-06T03-04-27.
    match = RUN_TIMESTAMP_RE.search(text)
    if match:
        stamp = match.group("stamp")
        if ":" not in stamp.split("T", 1)[1]:
            date_part, time_part = stamp.split("T", 1)
            time_part = time_part.replace("-", ":")
            text = f"{date_part}T{time_part}"

    try:
        return ensure_aware_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def iso_or_empty(value: datetime | None) -> str:
    if value is None:
        return ""

    return value.isoformat()


def seconds_between(started_at: datetime | None, finished_at: datetime | None) -> float | None:
    started_at = ensure_aware_utc(started_at)
    finished_at = ensure_aware_utc(finished_at)

    if started_at is None or finished_at is None:
        return None

    return max((finished_at - started_at).total_seconds(), 0.0)


def find_json_by_logical_name(run_dir: Path, logical_name: str) -> tuple[Path | None, Any | None]:
    ignored_parts = {"model_calls", "stage_executions"}
    matches: list[Path] = []

    for path in sorted(run_dir.rglob("*.json")):
        if ignored_parts.intersection(path.parts):
            continue

        if is_logical_artifact_name(normalize_stem(path), logical_name):
            matches.append(path)

    if not matches:
        return None, None

    # Return the latest numbered node artifact when a node ran multiple times.
    path = matches[-1]
    return path, read_json_safe(path)


def find_jsons_by_predicate(run_dir: Path, predicate) -> list[tuple[Path, Any]]:
    ignored_parts = {"model_calls", "stage_executions"}
    matches: list[tuple[Path, Any]] = []
    for path in sorted(run_dir.rglob("*.json")):
        if ignored_parts.intersection(path.parts):
            continue
        logical_name = normalize_stem(path)
        if predicate(logical_name):
            payload = read_json_safe(path)
            if payload is not None:
                matches.append((path, payload))
    return matches


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    except Exception:
        return []


def merge_csv_rows_by_key(
        primary_rows: list[dict[str, Any]],
        secondary_rows: list[dict[str, Any]],
        key: str,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for row in primary_rows:
        row_key = str(row.get(key) or "")
        if not row_key:
            continue
        merged[row_key] = dict(row)

    for row in secondary_rows:
        row_key = str(row.get(key) or "")
        if not row_key:
            continue
        merged.setdefault(row_key, {}).update({k: v for k, v in row.items() if v not in (None, "")})

    return [merged[item] for item in sorted(merged.keys(), key=lambda value: int(value) if value.isdigit() else value)]


def int_cell(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def float_cell(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def flatten_scalar_dict(prefix: str, value: Any, max_keys: int = 40) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    flattened: dict[str, Any] = {}
    added = 0

    for key, item in value.items():
        if added >= max_keys:
            break

        if isinstance(item, (str, int, float, bool)) or item is None:
            flattened[f"{prefix}_{key}"] = item
            added += 1

    return flattened


def summarize_file_list(run_dir: Path) -> tuple[list[str], int]:
    files: list[str] = []
    total_size = 0

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue

        try:
            relative_path = path.relative_to(run_dir).as_posix()
        except ValueError:
            relative_path = path.as_posix()

        files.append(relative_path)

        try:
            total_size += path.stat().st_size
        except OSError:
            pass

    return files, total_size


def collect_model_calls(run_dir: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    # Current analysis CSV format. Detailed JSON logs may also exist, but the CSV
    # is the canonical scalar source for global analysis.
    model_calls_csv = run_dir / "model_calls.csv"
    for row in read_csv_rows(model_calls_csv):
        payload: dict[str, Any] = dict(row)
        payload["_artifact_path"] = model_calls_csv.relative_to(run_dir).as_posix()
        payload["token_usage"] = {
            "prompt_tokens": int_cell(row.get("prompt_tokens")),
            "completion_tokens": int_cell(row.get("completion_tokens")),
            "total_tokens": int_cell(row.get("total_tokens")),
        }
        calls.append(payload)

    if calls:
        return calls

    # Legacy detailed JSON format. Still supported for older runs.
    model_calls_dir = run_dir / "model_calls"

    if model_calls_dir.exists():
        for path in sorted(model_calls_dir.glob("*.json")):
            payload = read_json_safe(path)

            if isinstance(payload, dict):
                payload["_artifact_path"] = path.relative_to(run_dir).as_posix()
                calls.append(payload)

    if calls:
        return calls

    # Legacy split CSV format from previous implementation.
    token_rows = read_csv_rows(run_dir / "model_call_tokens.csv")
    timing_rows = read_csv_rows(run_dir / "model_call_timings.csv")
    for row in merge_csv_rows_by_key(token_rows, timing_rows, "call_index"):
        payload = dict(row)
        payload["_artifact_path"] = "model_call_tokens.csv+model_call_timings.csv"
        payload["token_usage"] = {
            "prompt_tokens": int_cell(row.get("prompt_tokens")),
            "completion_tokens": int_cell(row.get("completion_tokens")),
            "total_tokens": int_cell(row.get("total_tokens")),
        }
        calls.append(payload)

    # Legacy format from older runs. New code should not create this file anymore.
    if calls:
        return calls

    legacy_path = run_dir / "artifacts" / "model_call_logs.json"
    legacy_payload = read_json_safe(legacy_path)

    if isinstance(legacy_payload, list):
        for index, payload in enumerate(legacy_payload, start=1):
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["_artifact_path"] = f"{legacy_path.relative_to(run_dir).as_posix()}#{index}"
                calls.append(payload)

    return calls


def collect_stage_logs(run_dir: Path) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []

    # Current format: one compact CSV with both timing and token data.
    stages_path = run_dir / "stages.csv"
    for row in read_csv_rows(stages_path):
        payload: dict[str, Any] = dict(row)
        payload["_artifact_path"] = stages_path.relative_to(run_dir).as_posix()
        payload["token_usage"] = {
            "prompt_tokens": int_cell(row.get("prompt_tokens")),
            "completion_tokens": int_cell(row.get("completion_tokens")),
            "total_tokens": int_cell(row.get("total_tokens")),
        }
        logs.append(payload)

    if logs:
        return logs

    # Legacy split CSV format from previous implementation.
    timing_rows = read_csv_rows(run_dir / "stage_timings.csv")
    token_rows = read_csv_rows(run_dir / "stage_tokens.csv")
    for row in merge_csv_rows_by_key(timing_rows, token_rows, "execution_index"):
        payload = dict(row)
        payload["_artifact_path"] = "stage_timings.csv+stage_tokens.csv"
        payload["token_usage"] = {
            "prompt_tokens": int_cell(row.get("prompt_tokens")),
            "completion_tokens": int_cell(row.get("completion_tokens")),
            "total_tokens": int_cell(row.get("total_tokens")),
        }
        logs.append(payload)

    if logs:
        return logs

    # Legacy JSON-only format from older runs. New code should not create these
    # files because they duplicate stages.csv and do not contain stage outputs.
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        for path in sorted(artifacts_dir.rglob("*stage_execution*.json")):
            payload = read_json_safe(path)

            if isinstance(payload, dict):
                payload["_artifact_path"] = path.relative_to(run_dir).as_posix()
                logs.append(payload)

    if logs:
        return logs

    stage_dir = run_dir / "stage_executions"
    if stage_dir.exists():
        for path in sorted(stage_dir.glob("*.json")):
            payload = read_json_safe(path)

            if isinstance(payload, dict):
                payload["_artifact_path"] = path.relative_to(run_dir).as_posix()
                logs.append(payload)

    return logs


def collect_errors(run_dir: Path) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    for path in sorted((run_dir / "errors").glob("*")) if (run_dir / "errors").exists() else []:
        if not path.is_file():
            continue

        errors.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "name": path.name,
                "text": read_text_safe(path),
            }
        )

    return errors


def sum_model_call_tokens(model_calls: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for call in model_calls:
        usage = call.get("token_usage")

        if not isinstance(usage, dict):
            continue

        for key in totals:
            try:
                totals[key] += int(usage.get(key) or 0)
            except (TypeError, ValueError):
                pass

    return totals


def sum_model_call_duration(model_calls: list[dict[str, Any]]) -> float:
    total = 0.0

    for call in model_calls:
        value = call.get("duration_seconds")

        if value is None:
            value = call.get("duration_ms")
            if value is not None:
                try:
                    value = float(value) / 1000.0
                except (TypeError, ValueError):
                    value = None

        try:
            total += float(value or 0.0)
        except (TypeError, ValueError):
            pass

    return total


def unique_values(items: list[dict[str, Any]], key: str) -> list[str]:
    values = []

    for item in items:
        value = item.get(key)

        if value is None:
            continue

        text = str(value)

        if text and text not in values:
            values.append(text)

    return values


def extract_spec_validation(run_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    preferred_names = [
        "spec_validation",
        "vega_lite_validated_spec",
    ]

    for name in preferred_names:
        path, payload = find_json_by_logical_name(run_dir, name)

        if isinstance(payload, dict):
            return payload, path

    return None, None


def extract_chart_metadata(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}

    _, candidate_set = find_json_by_logical_name(run_dir, "candidate_spec_set")
    if isinstance(candidate_set, dict):
        selected = (
                candidate_set.get("selected_candidate_spec")
                or candidate_set.get("selected")
                or candidate_set.get("selected_candidate")
        )

        if isinstance(selected, dict):
            row["selected_spec_id"] = selected.get("spec_id") or selected.get("id")
            row["selected_chart_family"] = selected.get("chart_family")
            row["selected_candidate_score"] = selected.get("score")
            row["selected_field_mapping_json"] = dump_json_cell(selected.get("field_mapping"))
            row["selected_encoding_roles_json"] = dump_json_cell(selected.get("encoding_roles"))

    vega_spec = None
    for logical_name in ("vega_spec", "vega_spec_raw"):
        _, candidate_payload = find_json_by_logical_name(run_dir, logical_name)
        if isinstance(candidate_payload, dict):
            vega_spec = candidate_payload
            row["vega_spec_artifact_name"] = logical_name
            break
    if isinstance(vega_spec, dict):
        if isinstance(vega_spec.get("spec"), dict):
            spec_json = vega_spec["spec"]
        elif isinstance(vega_spec.get("spec_json"), dict):
            spec_json = vega_spec["spec_json"]
        else:
            spec_json = vega_spec

        if isinstance(spec_json, dict):
            row["vega_mark"] = spec_json.get("mark")
            encoding = spec_json.get("encoding")
            if isinstance(encoding, dict):
                row["vega_encoding_channels"] = ",".join(sorted(encoding.keys()))
                fields = extract_spec_fields(spec_json)
            row["vega_field_count"] = len(fields)

    return row


def extract_metrics(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}
    metric_names = [
        "structural_spec_metric",
        "evaluation_summary",
    ]

    for name in metric_names:
        _, payload = find_json_by_logical_name(run_dir, name)

        if payload is None:
            continue

        row[f"has_{name}"] = True
        if isinstance(payload, dict):
            row.update(flatten_scalar_dict(name, payload))
    for name in metric_names:
        row.setdefault(f"has_{name}", False)

    return row


def extract_input_metadata(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}

    run_status = read_json_safe(run_dir / "run_status.json")
    if isinstance(run_status, dict):
        if run_status.get("query"):
            row["query"] = str(run_status.get("query"))
        if run_status.get("data_path"):
            row["data_path"] = str(run_status.get("data_path"))
        if run_status.get("error_type"):
            row["run_error_type"] = str(run_status.get("error_type"))
        if run_status.get("error"):
            row["run_error_summary"] = str(run_status.get("error"))[:500]
        if run_status.get("semantic_status") is not None:
            row["run_semantic_status"] = str(run_status.get("semantic_status"))

    query_path = run_dir / "input" / "query.txt"
    if "query" not in row:
        query = read_text_safe(query_path)
        if query is not None:
            row["query"] = query.strip()

    return row


def extract_data_profile(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}

    _, payload = find_json_by_logical_name(run_dir, "data_profile")
    if not isinstance(payload, dict):
        return row

    row["data_profile_row_count"] = payload.get("row_count")
    row["data_profile_col_count"] = payload.get("column_count") or payload.get("col_count")
    row["data_profile_status"] = payload.get("profile_status")
    row["data_profile_complexity"] = payload.get("data_complexity")
    row["data_profile_column_errors_count"] = len(payload.get("column_errors") or []) if isinstance(
        payload.get("column_errors"), list) else 0
    row["data_profile_quality_notes_count"] = len(payload.get("quality_notes") or []) if isinstance(
        payload.get("quality_notes"), list) else 0
    
    columns = payload.get("columns")
    if isinstance(columns, list):
        original_columns: list[str] = []
        safe_columns: list[str] = []
        renamed = 0
        for item in columns:
            if not isinstance(item, dict):
                continue
            original = str(item.get("original_name") or item.get("name") or "")
            safe = str(item.get("safe_name") or original)
            if original:
                original_columns.append(original)
            if safe:
                safe_columns.append(safe)
            if original and safe and original != safe:
                renamed += 1
        row["data_profile_renamed_column_count"] = renamed

    return row


def extract_spec_fields(spec_json: Any) -> list[str]:
    fields: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "field" and isinstance(item, str):
                    if item not in fields:
                        fields.append(item)
                elif key in {"fields", "groupby"} and isinstance(item, list):
                    for entry in item:
                        if isinstance(entry, str) and entry not in fields:
                            fields.append(entry)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(spec_json)
    return fields


def sum_stage_tokens(stage_logs: list[dict[str, Any]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for log in stage_logs:
        usage = log.get("token_usage")
        if isinstance(usage, dict):
            for key in totals:
                try:
                    totals[key] += int(usage.get(key) or 0)
                except (TypeError, ValueError):
                    pass
    return totals


def sum_stage_duration(stage_logs: list[dict[str, Any]]) -> float:
    total = 0.0
    for log in stage_logs:
        value = log.get("duration_seconds")
        if value in (None, ""):
            value = log.get("duration_ms")
            if value not in (None, ""):
                try:
                    value = float(value) / 1000.0
                except (TypeError, ValueError):
                    value = None
        try:
            total += float(value or 0.0)
        except (TypeError, ValueError):
            pass
    return total


def extract_data_preparation(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}

    _, payload = find_json_by_logical_name(run_dir, "data_preparation")
    if isinstance(payload, dict):
        column_name_map = payload.get("column_name_map") if isinstance(payload.get("column_name_map"), dict) else {}
        reverse_column_name_map = payload.get("reverse_column_name_map") if isinstance(
            payload.get("reverse_column_name_map"), dict) else {}
        original_columns = payload.get("original_columns") if isinstance(payload.get("original_columns"), list) else []
        safe_columns = payload.get("safe_columns") if isinstance(payload.get("safe_columns"), list) else []
        renamed_column_count = payload.get("renamed_column_count")
        if renamed_column_count is None and column_name_map:
            renamed_column_count = sum(1 for original, safe in column_name_map.items() if original != safe)

        row["prepared_data_path"] = payload.get("output_path")
        row["prepared_row_count"] = payload.get("row_count")
        row["prepared_col_count"] = payload.get("col_count")
        row["data_preparation_operations_json"] = dump_json_cell(payload.get("operations"))
        row["uses_safe_column_mapping"] = bool(renamed_column_count)
        row["safe_column_mapping_count"] = len(column_name_map)
        row["renamed_column_count"] = renamed_column_count or 0

    return row


def extract_spec_generation(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}
    _, payload = find_json_by_logical_name(run_dir, "vega_spec")
    if isinstance(payload, dict):
        spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
        warnings = payload.get("generation_warnings") if isinstance(payload.get("generation_warnings"), list) else []
        row["has_spec_generation_result"] = True
        row["spec_generation_backend"] = payload.get("generation_backend") or ""
        row["spec_generation_warning_count"] = len(warnings)
        row["spec_generation_has_explanation"] = bool(payload.get("generation_explanation"))
        row["spec_generation_spec_field_count"] = len(extract_spec_fields(spec))
        row["spec_generation_attempt_count"] = 1
        row["spec_generation_pipeline_attempt_count"] = 1
        row["spec_generation_response_attempt_count"] = 1
        row["spec_generation_failed_attempt_count"] = 0
        row["vega_mark"] = spec.get("mark") if isinstance(spec.get("mark"), str) else "composition" if any(key in spec for key in ("layer", "facet", "repeat", "concat", "hconcat", "vconcat")) else ""
        encoding = spec.get("encoding") if isinstance(spec.get("encoding"), dict) else {}
        row["vega_encoding_channels"] = ",".join(sorted(encoding.keys()))
        row["vega_field_count"] = len(extract_spec_fields(spec))
        return row

    # Legacy reader kept so global analysis can still summarize older runs.
    attempt_results = find_jsons_by_predicate(
        run_dir,
        lambda name: name.startswith("spec_generation_attempt_") and name.endswith("_result"),
    )
    if not attempt_results:
        result_path, legacy = find_json_by_logical_name(run_dir, "spec_generation_result")
        if isinstance(legacy, dict):
            attempt_results = [(result_path, legacy)] if result_path is not None else []
    if not attempt_results:
        row["has_spec_generation_result"] = False
        return row
    row["has_spec_generation_result"] = True
    row["spec_generation_attempt_count"] = len(attempt_results)
    row["spec_generation_pipeline_attempt_count"] = len(attempt_results)
    all_attempt_payloads = [item for _, item in attempt_results if isinstance(item, dict)]
    payload = all_attempt_payloads[-1] if all_attempt_payloads else {}
    attempts = payload.get("attempts") if isinstance(payload.get("attempts"), list) else []
    warnings = payload.get("warning_messages") if isinstance(payload.get("warning_messages"), list) else []
    spec = payload.get("spec_without_runtime_data") if isinstance(payload.get("spec_without_runtime_data"), dict) else payload.get("spec_json", {})
    response_attempt_count = sum(len(item.get("attempts") or []) for item in all_attempt_payloads)
    failed_response_attempt_count = sum(
        1
        for item in all_attempt_payloads
        for attempt in (item.get("attempts") or [])
        if isinstance(attempt, dict) and str(attempt.get("status") or "").lower() == "failed"
    )
    row["spec_generation_backend"] = payload.get("backend_name") or payload.get("generation_backend") or ""
    row["spec_generation_prompt_version"] = payload.get("prompt_version") or ""
    row["spec_generation_response_attempt_count"] = response_attempt_count or len(attempts)
    row["spec_generation_failed_attempt_count"] = failed_response_attempt_count
    row["spec_generation_warning_count"] = len(warnings)
    row["spec_generation_used_visrag_context"] = bool(payload.get("used_visrag_context"))
    row["spec_generation_has_explanation"] = bool(payload.get("explanation") or payload.get("generation_explanation"))
    row["spec_generation_final_generation_attempt_number"] = payload.get("generation_attempt_number") or ""
    row["spec_generation_max_generation_attempts"] = payload.get("max_generation_attempts") or ""
    row["spec_generation_spec_field_count"] = len(extract_spec_fields(spec if isinstance(spec, dict) else {}))
    if isinstance(spec, dict):
        row["vega_mark"] = spec.get("mark") if isinstance(spec.get("mark"), str) else "composition" if any(key in spec for key in ("layer", "facet", "repeat", "concat", "hconcat", "vconcat")) else ""
        encoding = spec.get("encoding") if isinstance(spec.get("encoding"), dict) else {}
        row["vega_encoding_channels"] = ",".join(sorted(encoding.keys()))
        row["vega_field_count"] = len(extract_spec_fields(spec))
    return row

def extract_spec_validation_retry(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}
    summary_path, summary = find_json_by_logical_name(run_dir, "spec_validation_retry_summary")
    attempts = summary.get("attempts") if isinstance(summary, dict) and isinstance(summary.get("attempts"),
                                                                                   list) else []

    row["has_spec_validation_retry_summary"] = isinstance(summary, dict)
    if not isinstance(summary, dict):
        return row

    row["spec_validation_retry_summary_path"] = summary_path.relative_to(run_dir).as_posix() if summary_path else ""
    row["spec_validation_retry_succeeded"] = bool(summary.get("succeeded"))
    row["spec_validation_generation_attempt_count"] = summary.get("completed_generation_attempts") or len(attempts)
    row["spec_validation_max_generation_attempts"] = summary.get("max_generation_attempts") or ""

    valid_flags = [bool(item.get("is_valid")) for item in attempts if isinstance(item, dict)]
    row["spec_validation_attempt_valid_flags_json"] = dump_json_cell(valid_flags)

    final_attempt = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
    final_errors = final_attempt.get("validation_errors") if isinstance(final_attempt.get("validation_errors"),
                                                                        list) else []
    final_hints = final_attempt.get("repair_hints") if isinstance(final_attempt.get("repair_hints"), list) else []
    row["spec_validation_final_error_count"] = len(final_errors)
    row["spec_validation_final_repair_hint_count"] = len(final_hints)
    reports = [
        path.relative_to(run_dir).as_posix()
        for path in sorted(run_dir.rglob("*spec_validation_attempt_*_report.md"))
    ]
    row["spec_validation_attempt_report_paths_json"] = dump_json_cell(reports)
    return row


def extract_semantic_feedback(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}
    summary_path, summary = find_json_by_logical_name(run_dir, "semantic_feedback_loop_summary")
    row["has_semantic_feedback_loop_summary"] = isinstance(summary, dict)
    if isinstance(summary, dict):
        row["semantic_feedback_summary_path"] = summary_path.relative_to(run_dir).as_posix() if summary_path else ""
        row["semantic_loop_enabled"] = bool(summary.get("enabled"))
        row["semantic_attempt_count"] = summary.get("attempt_count") or 0
        row["semantic_retry_count"] = summary.get("retry_count") or 0
        row["semantic_final_status"] = summary.get("final_status") or ""
        row["semantic_accepted"] = bool(summary.get("accepted"))
        row["semantic_accepted_attempt"] = summary.get("accepted_attempt") or ""
        row["semantic_final_confidence"] = summary.get("final_confidence") or 0
        row["semantic_saved_feedback_count"] = summary.get("saved_feedback_count") or 0
        row["semantic_missing_requirements_json"] = dump_json_cell(summary.get("missing_requirements") or [])
        row["semantic_improvement_comments_json"] = dump_json_cell(summary.get("improvement_comments") or [])
        row["semantic_feedback_corpus_path"] = summary.get("feedback_corpus_path") or ""
    else:
        row["semantic_loop_enabled"] = False
        row["semantic_attempt_count"] = 0
        row["semantic_retry_count"] = 0
        row["semantic_final_status"] = ""
        row["semantic_accepted"] = False
        row["semantic_accepted_attempt"] = ""
        row["semantic_final_confidence"] = ""
        row["semantic_saved_feedback_count"] = 0

    descriptions = find_jsons_by_predicate(run_dir, lambda name: "semantic_attempt_" in name and name.endswith(
        "vlm_chart_description"))
    facts = find_jsons_by_predicate(run_dir,
                                    lambda name: "semantic_attempt_" in name and name.endswith("chart_fact_summary"))
    judges = find_jsons_by_predicate(run_dir,
                                     lambda name: "semantic_attempt_" in name and name.endswith("answer_judge"))
    examples = find_jsons_by_predicate(run_dir,
                                       lambda name: "semantic_attempt_" in name and name.endswith("feedback_example"))

    row["semantic_vlm_description_count"] = len(descriptions)
    row["semantic_chart_fact_summary_count"] = len(facts)
    row["semantic_answer_judge_count"] = len(judges)
    row["semantic_feedback_example_count"] = len(examples)
    if judges and isinstance(judges[-1][1], dict):
        final = judges[-1][1]
        row["semantic_final_answers_user_query"] = bool(final.get("answers_user_query"))
        row["semantic_final_retry_recommendation"] = final.get("retry_recommendation") or ""
        row["semantic_final_missing_requirements_json"] = dump_json_cell(final.get("missing_requirements") or [])
        row["semantic_final_improvement_comments_json"] = dump_json_cell(final.get("improvement_comments") or [])

    return row


def collect_run(run_dir: Path, artifacts_root: Path) -> dict[str, Any]:
    files, total_size_bytes = summarize_file_list(run_dir)
    model_calls = collect_model_calls(run_dir)
    stage_logs = collect_stage_logs(run_dir)
    errors = collect_errors(run_dir)

    file_paths = [run_dir / file for file in files]
    png_paths = [
        path
        for path in file_paths
        if path.suffix.lower() == ".png"
    ]
    json_paths = [
        path
        for path in file_paths
        if path.suffix.lower() == ".json"
    ]
    csv_paths = [
        path
        for path in file_paths
        if path.suffix.lower() == ".csv"
    ]

    started_candidates = [
        parse_datetime_safe(log.get("started_at"))
        for log in stage_logs
    ]
    finished_candidates = [
        parse_datetime_safe(log.get("finished_at"))
        for log in stage_logs
    ]
    started_candidates = [item for item in started_candidates if item is not None]
    finished_candidates = [item for item in finished_candidates if item is not None]

    run_started_at = min(started_candidates) if started_candidates else parse_datetime_safe(run_dir.name)
    run_finished_at = max(finished_candidates) if finished_candidates else None

    if run_finished_at is None:
        try:
            run_finished_at = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
        except OSError:
            run_finished_at = None

    duration_seconds = seconds_between(run_started_at, run_finished_at)

    stage_statuses = [str(log.get("status") or "").lower() for log in stage_logs]
    stage_names = [
        str(log.get("pipeline_stage") or log.get("node_name") or "")
        for log in stage_logs
        if log.get("pipeline_stage") or log.get("node_name")
    ]

    failed_stage_logs = [
        log
        for log in stage_logs
        if str(log.get("status") or "").lower() == "failed"
    ]
    skipped_stage_logs = [
        log
        for log in stage_logs
        if str(log.get("status") or "").lower() == "skipped"
    ]
    succeeded_stage_logs = [
        log
        for log in stage_logs
        if str(log.get("status") or "").lower() == "succeeded"
    ]

    stage_token_totals = sum_stage_tokens(stage_logs)
    stage_duration = sum_stage_duration(stage_logs)

    token_totals = sum_model_call_tokens(model_calls)
    model_call_duration = sum_model_call_duration(model_calls)

    spec_validation, spec_validation_path = extract_spec_validation(run_dir)
    spec_is_valid = None
    spec_validation_error_count = None

    if isinstance(spec_validation, dict):
        spec_is_valid = spec_validation.get("is_valid")
        validation_errors = spec_validation.get("validation_errors")
        if isinstance(validation_errors, list):
            spec_validation_error_count = len(validation_errors)

    png_relative_paths = [
        path.relative_to(run_dir).as_posix()
        for path in png_paths
    ]

    plot_png_paths = [
        path for path in png_relative_paths
        if "plot" in Path(path).stem.lower()
    ]

    has_png = bool(png_paths)
    has_plot_png = bool(plot_png_paths)

    row: OrderedDict[str, Any] = OrderedDict()
    row["run_id"] = run_dir.name
    row["run_path"] = run_dir.relative_to(artifacts_root).as_posix()
    row["run_path_abs"] = run_dir.resolve().as_posix()
    row["started_at"] = iso_or_empty(run_started_at)
    row["finished_at"] = iso_or_empty(run_finished_at)
    row["duration_seconds"] = round(duration_seconds, 6) if duration_seconds is not None else ""

    row["status"] = "unknown"
    if failed_stage_logs or errors:
        row["status"] = "failed"
    elif any(name == "completed" for name in stage_names) or stage_logs:
        row["status"] = "succeeded"

    row["is_success"] = row["status"] == "succeeded"
    row["has_errors"] = bool(errors)
    row["error_count"] = len(errors)
    row["first_error"] = errors[0]["text"][:500] if errors and errors[0].get("text") else ""

    row["stage_count"] = len(stage_logs)
    row["stage_succeeded_count"] = len(succeeded_stage_logs)
    row["stage_failed_count"] = len(failed_stage_logs)
    row["stage_skipped_count"] = len(skipped_stage_logs)
    row["stage_duration_seconds"] = round(stage_duration, 6)
    row["stage_prompt_tokens"] = stage_token_totals["prompt_tokens"]
    row["stage_completion_tokens"] = stage_token_totals["completion_tokens"]
    row["stage_total_tokens"] = stage_token_totals["total_tokens"]
    row["failed_stage"] = (
        failed_stage_logs[0].get("pipeline_stage")
        or failed_stage_logs[0].get("node_name")
        if failed_stage_logs
        else ""
    )
    row["has_stages_csv"] = (run_dir / "stages.csv").exists()
    row["stages_csv_path"] = "stages.csv" if (run_dir / "stages.csv").exists() else ""
    row["has_legacy_stage_split_csv"] = (run_dir / "stage_timings.csv").exists() or (
                run_dir / "stage_tokens.csv").exists()
    row["has_legacy_stage_execution_json"] = any("stage_execution" in str(path) for path in files)

    row["model_call_count"] = len(model_calls)
    row["model_call_prompt_tokens"] = token_totals["prompt_tokens"]
    row["model_call_completion_tokens"] = token_totals["completion_tokens"]
    row["model_call_total_tokens"] = token_totals["total_tokens"]
    row["model_call_duration_seconds"] = round(model_call_duration, 6)
    row["model_call_error_count"] = sum(
        1
        for call in model_calls
        if call.get("parser_errors") or call.get("error")
    )
    row["has_model_calls_csv"] = (run_dir / "model_calls.csv").exists()
    row["model_calls_csv_path"] = "model_calls.csv" if (run_dir / "model_calls.csv").exists() else ""
    row["has_legacy_model_call_split_csv"] = (run_dir / "model_call_tokens.csv").exists() or (
                run_dir / "model_call_timings.csv").exists()

    row["has_png"] = has_png
    row["has_plot_png"] = has_plot_png
    row["png_count"] = len(png_paths)
    row["plot_png_path"] = plot_png_paths[0] if plot_png_paths else ""

    row["has_spec_validation"] = spec_validation is not None
    row["spec_is_valid"] = spec_is_valid if spec_is_valid is not None else ""
    row["spec_validation_error_count"] = spec_validation_error_count if spec_validation_error_count is not None else ""
    row["spec_validation_path"] = (
        spec_validation_path.relative_to(run_dir).as_posix()
        if spec_validation_path
        else ""
    )

    row["json_file_count"] = len(json_paths)
    row["csv_file_count"] = len(csv_paths)
    row["total_file_count"] = len(files)
    row["total_size_bytes"] = total_size_bytes

    row.update(extract_input_metadata(run_dir))
    row.update(extract_data_profile(run_dir))
    row.update(extract_data_preparation(run_dir))
    row.update(extract_spec_generation(run_dir))
    row.update(extract_spec_validation_retry(run_dir))
    row.update(extract_semantic_feedback(run_dir))
    row.update(extract_chart_metadata(run_dir))
    row.update(extract_metrics(run_dir))

    return row


def collect_runs(artifacts_root: Path) -> list[dict[str, Any]]:
    if not artifacts_root.exists():
        raise FileNotFoundError(f"Artifacts root does not exist: {artifacts_root}")

    if not artifacts_root.is_dir():
        raise NotADirectoryError(f"Artifacts root is not a directory: {artifacts_root}")

    rows: list[dict[str, Any]] = []

    for run_dir in sorted(artifacts_root.iterdir()):
        if not run_dir.is_dir():
            continue

        if run_dir.name.startswith("."):
            continue

        # Only first-level directories are treated as pipeline runs.
        rows.append(collect_run(run_dir=run_dir, artifacts_root=artifacts_root))

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_columns = [
        "run_id",
        "started_at",
        "finished_at",
        "duration_seconds",
        "status",
        "is_success",
        "run_error_type",
        "run_error_summary",
        "has_errors",
        "error_count",
        "first_error",
        "stage_count",
        "stage_succeeded_count",
        "stage_failed_count",
        "stage_skipped_count",
        "stage_duration_seconds",
        "stage_prompt_tokens",
        "stage_completion_tokens",
        "stage_total_tokens",
        "model_call_count",
        "model_call_prompt_tokens",
        "model_call_completion_tokens",
        "model_call_total_tokens",
        "model_call_duration_seconds",
        "model_call_error_count",
        "has_plot_png",
        "png_count",
        "data_profile_row_count",
        "data_profile_col_count",
        "data_profile_status",
        "data_profile_complexity",
        "data_profile_quality_notes_count",
        "data_profile_column_errors_count",
        "data_profile_renamed_column_count",
        "prepared_row_count",
        "prepared_col_count",
        "renamed_column_count",
        "has_spec_generation_result",
        "spec_generation_backend",
        "spec_generation_warning_count",
        "spec_generation_has_explanation",
        "spec_generation_spec_field_count",
        "has_spec_validation",
        "spec_is_valid",
        "spec_validation_error_count",
        "vega_mark",
        "vega_encoding_channels",
        "vega_field_count",
        "has_semantic_feedback_loop_summary",
        "semantic_loop_enabled",
        "semantic_attempt_count",
        "semantic_retry_count",
        "semantic_final_status",
        "semantic_accepted",
        "semantic_accepted_attempt",
        "semantic_final_confidence",
        "semantic_saved_feedback_count",
        "semantic_final_answers_user_query",
        "semantic_final_retry_recommendation",
        "evaluation_summary_technical_status",
        "evaluation_summary_semantic_status",
        "evaluation_summary_chart_accepted",
        "evaluation_summary_semantic_retry_count",
        "evaluation_summary_technical_retry_count",
        "has_structural_spec_metric",
        "structural_spec_metric_score",
        "has_evaluation_summary",
        "json_file_count",
        "csv_file_count",
        "total_file_count",
        "total_size_bytes",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=base_columns, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all pipeline run folders from the global artifacts directory "
            "into a single analysis CSV."
        )
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Global artifacts directory. Default: artifacts",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: <artifacts-root>/global_analysis.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else artifacts_root / DEFAULT_OUTPUT_NAME
    )

    rows = collect_runs(artifacts_root)
    write_csv(rows, output_path)

    success_count = sum(1 for row in rows if row.get("is_success") is True)
    failed_count = sum(1 for row in rows if row.get("status") == "failed")

    print(f"Artifacts root: {artifacts_root}")
    print(f"Runs collected: {len(rows)}")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
