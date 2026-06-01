from __future__ import annotations

from pathlib import Path

from src.application.contracts import PipelineRequest, PipelineResult
from src.application.run_report import build_run_report, save_error_report, save_run_report, save_task_request
from src.application.settings import ViRAGESettings
from src.domain.models import (
    EmptyChartCheckResult,
    PlotImageArtifact,
    SpecValidationResult,
    StructuralSpecMetric,
    TokenUsage,
)
from src.infrastructure.runtime import RuntimeContext


def test_build_run_report_collects_core_metrics(tmp_path: Path) -> None:
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "run"
    runtime.ensure_run_dir()
    image_path = tmp_path / "artifacts" / "run" / "chart.png"
    image_path.write_bytes(b"png")
    spec_path = runtime.save_json_artifact("nodes/vega_spec.json", {"mark": "bar"}, run_id="run")
    request = PipelineRequest(query="compare values", data_path="data.csv", run_id="run")
    result = PipelineResult(
        run_id="run",
        query="compare values",
        data_path="data.csv",
        spec_validation=SpecValidationResult(is_valid=True, validated_spec={"mark": "bar"}),
        empty_chart_check=EmptyChartCheckResult(empty_chart_signal=False, empty_chart_status="non_empty"),
        plot_image=PlotImageArtifact(image_path=image_path.as_posix()).model_dump(),
        structural_spec_metric=StructuralSpecMetric(score=0.82),
        token_usage_summary=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        artifact_paths={"vega_spec": spec_path},
    )

    report = build_run_report(request=request, result=result, runtime=runtime, status="completed")

    assert report["run_id"] == "run"
    assert report["status"] == "completed"
    assert report["has_plot"] is True
    assert report["generated_image_exists"] is True
    assert report["generated_spec_exists"] is True
    assert report["valid_spec"] is True
    assert report["empty_chart"] is False
    assert report["spec_score"] == 0.82
    assert report["prompt_tokens"] == 10
    assert report["completion_tokens"] == 5
    assert report["total_tokens"] == 15
    assert report["metrics"]["spec_score"] == 0.82


def test_run_report_files_are_saved(tmp_path: Path) -> None:
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "run"
    request = PipelineRequest(query="show distribution", data_path="data.csv", run_id="run")
    result = PipelineResult(run_id="run", query=request.query, data_path=request.data_path)

    task_path = save_task_request(runtime, request)
    report_path = save_run_report(runtime, request=request, result=result, status="completed")

    assert Path(task_path).exists()
    assert Path(report_path).exists()
    assert Path(report_path).name == "run_report.json"


def test_error_report_and_failed_run_report_are_saved(tmp_path: Path) -> None:
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    runtime.current_run_id = "failed"
    request = PipelineRequest(query="bad", data_path="missing.csv", run_id="failed")

    errors_path = save_error_report(
        runtime,
        request=request,
        error_type="failed",
        error="RuntimeError: boom",
        traceback_text="traceback",
    )
    report_path = save_run_report(
        runtime,
        request=request,
        result=None,
        status="failed",
        error_type="failed",
        error="RuntimeError: boom",
    )

    assert Path(errors_path).exists()
    assert Path(report_path).exists()
    assert "RuntimeError: boom" in Path(report_path).read_text(encoding="utf-8")
