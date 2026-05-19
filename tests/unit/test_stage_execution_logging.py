from __future__ import annotations

import csv
from pathlib import Path

from src.application.settings import ViRAGESettings
from src.domain.models import ModelCallLog, StageExecutionLog, TokenUsage
from src.infrastructure.runtime import RuntimeContext


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_stage_execution_log_is_saved_to_single_stages_csv(tmp_path: Path) -> None:
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
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    runtime.add_stage_execution_log(log)

    run_dir = tmp_path / "run-1"
    rows = read_csv_rows(run_dir / "stages.csv")

    assert len(rows) == 1
    assert rows[0]["node_name"] == "chart_generator"
    assert rows[0]["pipeline_stage"] == "chart_generation"
    assert rows[0]["total_tokens"] == "3"
    assert rows[0]["duration_seconds"] == "1.0"
    assert not (run_dir / "stage_executions").exists()
    assert not (run_dir / "stage_timings.csv").exists()
    assert not (run_dir / "stage_tokens.csv").exists()
    assert not sorted((run_dir / "artifacts").glob("*stage_execution*.json"))


def test_model_call_logs_are_saved_to_single_model_calls_csv(tmp_path: Path) -> None:
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path))
    runtime.current_run_id = "run-1"
    runtime.reset_artifact_indices(run_id="run-1")

    runtime.add_model_call_log(
        ModelCallLog(
            stage="query_request_analysis",
            model_role="reasoning",
            model_name="test-model",
            prompt="prompt",
            raw_response="{}",
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
            duration_ms=50.0,
            duration_seconds=0.05,
        )
    )
    runtime.save_model_log_artifacts(run_id="run-1")

    run_dir = tmp_path / "run-1"
    rows = read_csv_rows(run_dir / "model_calls.csv")

    assert (run_dir / "model_calls" / "001_query_request_analysis_reasoning-01.json").exists()
    assert len(rows) == 1
    assert rows[0]["stage"] == "query_request_analysis"
    assert rows[0]["total_tokens"] == "3"
    assert rows[0]["duration_seconds"] == "0.05"
    assert not (run_dir / "model_call_tokens.csv").exists()
    assert not (run_dir / "model_call_timings.csv").exists()
    assert not (run_dir / "artifacts" / "model_call_logs.json").exists()
    assert not (run_dir / "artifacts" / "token_usage_summary.json").exists()


def test_wrapped_graph_node_records_stage_execution_log(tmp_path: Path) -> None:
    from src.domain.enums import PipelineStage
    from src.graph.builder import _wrap_stage_node

    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path))
    runtime.current_run_id = "run-1"
    runtime.reset_artifact_indices(run_id="run-1")

    def node(state):
        return {"stage": PipelineStage.DATA_PREPARATION, "value": 42}

    wrapped = _wrap_stage_node(name="data_preparation", callable_node=node, runtime=runtime)
    output = wrapped({"run_id": "run-1", "stage_execution_logs": []})

    assert output["value"] == 42
    assert len(runtime.stage_execution_logs) == 1
    log = runtime.stage_execution_logs[0]
    assert log.node_name == "data_preparation"
    assert log.pipeline_stage == PipelineStage.DATA_PREPARATION.value
    assert log.status == "succeeded"
    rows = read_csv_rows(tmp_path / "run-1" / "stages.csv")
    assert len(rows) == 1
    assert rows[0]["node_name"] == "data_preparation"
    assert not sorted((tmp_path / "run-1" / "artifacts").glob("*stage_execution*.json"))


def test_wrapped_graph_node_emits_started_step_before_execution(tmp_path: Path) -> None:
    from src.graph.builder import _wrap_stage_node

    emitted = []
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path))
    runtime.current_run_id = "run-1"
    runtime.step_callback = emitted.append

    def node(state):
        assert emitted
        assert emitted[-1].stage == "chart_generator"
        assert emitted[-1].summary == "Running now"
        return {"value": 1}

    wrapped = _wrap_stage_node(name="chart_generator", callable_node=node, runtime=runtime)
    wrapped({"run_id": "run-1", "stage_execution_logs": []})

    assert emitted[0].title == "Chart generation"
