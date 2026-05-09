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
    art = run / "artifacts"
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
