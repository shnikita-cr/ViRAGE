from __future__ import annotations

from pathlib import Path

from src.application.settings import ViRAGESettings
from src.domain.models import ModelCallLog, StageExecutionLog, TokenUsage
from src.infrastructure.runtime import RuntimeContext


def test_stage_execution_log_is_saved_under_artifacts_not_stage_executions(tmp_path: Path) -> None:
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path))
    runtime.current_run_id = "run-1"
    runtime.reset_artifact_indices(run_id="run-1")

    log = StageExecutionLog(
        node_name="chart_generator",
        pipeline_stage="chart_generation",
        status="succeeded",
        started_at="2026-05-06T00:00:00+00:00",
        finished_at="2026-05-06T00:00:01+00:00",
        duration_ms=1000.0,
        duration_seconds=1.0,
    )

    runtime.add_stage_execution_log(log)

    run_dir = tmp_path / "run-1"
    artifact_files = sorted((run_dir / "artifacts").glob("*_stage_execution_chart_generator_succeeded.json"))

    assert artifact_files
    assert not (run_dir / "stage_executions").exists()
    assert (run_dir / "stage_timings.csv").exists()
    assert (run_dir / "stage_tokens.csv").exists()


def test_model_call_logs_json_is_not_written_to_artifacts(tmp_path: Path) -> None:
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path))
    runtime.current_run_id = "run-1"
    runtime.reset_artifact_indices(run_id="run-1")

    runtime.add_model_call_log(
        ModelCallLog(
            stage="query_understanding",
            model_role="reasoning",
            model_name="test-model",
            prompt="prompt",
            raw_response="{}",
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )
    )
    runtime.save_model_log_artifacts(run_id="run-1")

    run_dir = tmp_path / "run-1"

    assert (run_dir / "model_calls" / "001_query_understanding_reasoning-01.json").exists()
    assert (run_dir / "model_call_tokens.csv").exists()
    assert (run_dir / "model_call_timings.csv").exists()
    assert not (run_dir / "artifacts" / "model_call_logs.json").exists()
    assert (run_dir / "artifacts" / "token_usage_summary.json").exists()
