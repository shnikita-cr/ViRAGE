from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.tools.build_global_artifacts_analysis import collect_run


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_global_analysis_collects_stages_csv_and_safe_mapping(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    run = root / "2026-05-09T00-00-00_test"
    art = run / "nodes"
    art.mkdir(parents=True)

    write_csv(
        run / "stages.csv",
        [
            {
                "execution_index": 1,
                "node_name": "data_preparation",
                "pipeline_stage": "data_preparation",
                "status": "succeeded",
                "started_at": "2026-05-09T00:00:00+00:00",
                "finished_at": "2026-05-09T00:00:01+00:00",
                "duration_seconds": 1.0,
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "model_call_count": 1,
                "model_call_start_index": 1,
                "model_call_end_index": 1,
                "error": "",
            }
        ],
    )
    (art / "002_data_preparation.json").write_text(
        json.dumps(
            {
                "output_path": "cleaned.csv",
                "row_count": 2,
                "col_count": 2,
                "column_name_map": {"Metric Value (%)": "Metric_Value"},
                "reverse_column_name_map": {"Metric_Value": "Metric Value (%)"},
                "original_columns": ["Metric Value (%)"],
                "safe_columns": ["Metric_Value"],
                "renamed_column_count": 1,
                "operations": ["safe_column_mapping:1"],
            }
        ),
        encoding="utf-8",
    )

    row = collect_run(run, root)

    assert row["stage_count"] == 1
    assert row["stage_total_tokens"] == 3
    assert row["stage_duration_seconds"] == 1.0
    assert row["has_stages_csv"] is True
    assert row["has_legacy_stage_split_csv"] is False
    assert row["has_legacy_stage_execution_json"] is False
    assert row["uses_safe_column_mapping"] is True
    assert row["renamed_column_count"] == 1
    assert row["safe_column_mapping_count"] == 1


def test_global_analysis_supports_legacy_split_stage_csv(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    run = root / "2026-05-09T00-00-00_legacy"
    run.mkdir(parents=True)

    write_csv(
        run / "stage_timings.csv",
        [
            {
                "execution_index": 1,
                "node_name": "visrag",
                "pipeline_stage": "visrag",
                "status": "succeeded",
                "duration_seconds": 2.0,
                "started_at": "2026-05-09T00:00:00+00:00",
                "finished_at": "2026-05-09T00:00:02+00:00",
                "model_call_count": 0,
                "error": "",
            }
        ],
    )
    write_csv(
        run / "stage_tokens.csv",
        [
            {
                "execution_index": 1,
                "node_name": "visrag",
                "pipeline_stage": "visrag",
                "status": "succeeded",
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "total_tokens": 9,
                "model_call_start_index": 0,
                "model_call_end_index": 0,
                "model_call_count": 0,
            }
        ],
    )

    row = collect_run(run, root)

    assert row["stage_count"] == 1
    assert row["stage_total_tokens"] == 9
    assert row["stage_duration_seconds"] == 2.0
    assert row["has_stages_csv"] is False
    assert row["has_legacy_stage_split_csv"] is True


def test_global_analysis_collects_spec_generation_result(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    run = root / "2026-05-09T00-00-00_codegen"
    art = run / "nodes"
    art.mkdir(parents=True)

    (art / "003_spec_generation_result.json").write_text(
        json.dumps(
            {
                "backend_name": "vegachat_codegen",
                "prompt_version": "vega_chat_v1",
                "spec_json": {"mark": "bar", "encoding": {"x": {"field": "Method"}}},
                "spec_without_runtime_data": {"mark": "bar", "encoding": {"x": {"field": "Method"}}},
                "explanation": "Bar chart.",
                "attempts": [
                    {"attempt_number": 1, "status": "succeeded", "raw_response": "..."}
                ],
                "warning_messages": ["model_output_missing_schema_added"],
                "artifact_paths": {
                    "spec_generation_prompt": "artifacts/001_spec_generation_prompt.txt",
                    "spec_generation_raw_response": "artifacts/002_spec_generation_raw_response.txt",
                },
                "used_visrag_context": True,
            }
        ),
        encoding="utf-8",
    )
    (art / "004_vega_spec_raw.json").write_text(
        json.dumps(
            {
                "spec_json": {"mark": "bar", "encoding": {"x": {"field": "Method"}}},
                "generation_backend": "vegachat_codegen",
            }
        ),
        encoding="utf-8",
    )

    row = collect_run(run, root)

    assert row["has_spec_generation_result"] is True
    assert row["spec_generation_backend"] == "vegachat_codegen"
    assert row["spec_generation_prompt_version"] == "vega_chat_v1"
    assert row["spec_generation_attempt_count"] == 1
    assert row["spec_generation_warning_count"] == 1
    assert row["spec_generation_used_visrag_context"] is True
    assert row["vega_mark"] == "bar"
    assert row["vega_field_count"] == 1


def test_global_analysis_collects_spec_generation_retry_attempts(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    run = root / "2026-05-09T00-00-00_retry"
    art = run / "nodes"
    art.mkdir(parents=True)

    for attempt_number in (1, 2):
        (art / f"00{attempt_number}_spec_generation_attempt_{attempt_number:03d}_result.json").write_text(
            json.dumps(
                {
                    "backend_name": "vegachat_codegen",
                    "prompt_version": "vega_chat_v1",
                    "spec_without_runtime_data": {"mark": "bar", "encoding": {"x": {"field": "Method"}}},
                    "explanation": "fixed" if attempt_number == 2 else "bad",
                    "attempts": [{"attempt_number": 1, "status": "succeeded"}],
                    "warning_messages": [],
                    "artifact_paths": {
                        f"spec_generation_attempt_{attempt_number:03d}_prompt": f"artifacts/prompt_{attempt_number}.txt",
                        f"spec_generation_attempt_{attempt_number:03d}_raw_response": f"artifacts/raw_{attempt_number}.txt",
                    },
                    "used_visrag_context": True,
                    "generation_attempt_number": attempt_number,
                    "max_generation_attempts": 2,
                    "previous_validation_errors": ["bad"] if attempt_number == 2 else [],
                    "previous_repair_hints": ["fix"] if attempt_number == 2 else [],
                }
            ),
            encoding="utf-8",
        )

    (art / "010_spec_validation_retry_summary.json").write_text(
        json.dumps(
            {
                "max_generation_attempts": 2,
                "completed_generation_attempts": 2,
                "succeeded": True,
                "attempts": [
                    {"is_valid": False, "validation_errors": ["bad"], "repair_hints": ["fix"]},
                    {"is_valid": True, "validation_errors": [], "repair_hints": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    (art / "009_spec_validation_attempt_001_report.md").write_text("report", encoding="utf-8")

    row = collect_run(run, root)

    assert row["spec_generation_attempt_count"] == 2
    assert row["spec_generation_warning_count"] == 0
