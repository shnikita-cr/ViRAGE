from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.application.contracts import PipelineRequest, PipelineResult
from src.infrastructure.runtime import RuntimeContext


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _path_from_plot_image(plot_image: Any) -> str | None:
    payload = _model_dump(plot_image)
    value = payload.get("image_path")
    return str(value) if value else None


def _path_exists(path: str | None) -> bool:
    return bool(path) and Path(str(path)).exists()


def _stage_duration_seconds(result: PipelineResult | None) -> float | None:
    if result is None:
        return None
    total = 0.0
    seen = False
    for log in result.stage_execution_logs:
        duration = _maybe_float(getattr(log, "duration_seconds", None))
        if duration is not None:
            total += duration
            seen = True
    return total if seen else None


def _semantic_status(result: PipelineResult | None) -> str | None:
    if result is None:
        return None
    if result.evaluation_summary is not None:
        value = getattr(result.evaluation_summary, "semantic_status", None)
        if value:
            return str(value)
    summary = result.semantic_feedback_loop_summary
    if summary is not None:
        value = getattr(summary, "final_status", None) or getattr(summary, "semantic_status", None)
        if value:
            return str(value)
    return None


def _vision_score(result: PipelineResult | None) -> float | None:
    if result is None:
        return None
    summary = result.evaluation_summary
    if summary is not None:
        scores = getattr(summary, "benchmark_scores", {}) or {}
        for key in ("vision_score", "vision_metric", "vision"):
            value = _maybe_float(scores.get(key))
            if value is not None:
                return value
    judge = result.visual_chart_judge
    if judge is not None:
        values = [
            _maybe_float(getattr(judge, "plot_area_usage_score", None)),
            _maybe_float(getattr(judge, "axis_domain_score", None)),
            _maybe_float(getattr(judge, "layout_compactness_score", None)),
            _maybe_float(getattr(judge, "repeat_axis_label_score", None)),
            _maybe_float(getattr(judge, "publication_layout_score", None)),
        ]
        clean = [value for value in values if value is not None]
        if clean:
            return sum(clean) / len(clean)
    return None


def _chart_quality_report(result: PipelineResult | None) -> dict[str, Any]:
    if result is None or result.plot_rendering is None:
        return {}
    report = getattr(result.plot_rendering, "chart_quality_report", {}) or {}
    return report if isinstance(report, dict) else {}


def _chart_quality_status(result: PipelineResult | None) -> str | None:
    report = _chart_quality_report(result)
    value = report.get("status")
    return str(value) if value else None


def _chart_quality_score(result: PipelineResult | None) -> float | None:
    report = _chart_quality_report(result)
    return _maybe_float(report.get("score"))


def _chart_quality_issue_counts(result: PipelineResult | None) -> dict[str, int]:
    report = _chart_quality_report(result)
    issues = report.get("issues") if isinstance(report, dict) else []
    if not isinstance(issues, list):
        return {"critical": 0, "warning": 0}
    critical = 0
    warning = 0
    for item in issues:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "").lower()
        critical += int(severity == "critical")
        warning += int(severity == "warning")
    return {"critical": critical, "warning": warning}


def _visual_publication_scores(result: PipelineResult | None) -> dict[str, float | None]:
    if result is None or result.visual_chart_judge is None:
        return {
            "plot_area_usage_score": None,
            "axis_domain_score": None,
            "layout_compactness_score": None,
            "repeat_axis_label_score": None,
            "publication_layout_score": None,
        }
    judge = result.visual_chart_judge
    return {
        "plot_area_usage_score": _maybe_float(getattr(judge, "plot_area_usage_score", None)),
        "axis_domain_score": _maybe_float(getattr(judge, "axis_domain_score", None)),
        "layout_compactness_score": _maybe_float(getattr(judge, "layout_compactness_score", None)),
        "repeat_axis_label_score": _maybe_float(getattr(judge, "repeat_axis_label_score", None)),
        "publication_layout_score": _maybe_float(getattr(judge, "publication_layout_score", None)),
    }


def _spec_score(result: PipelineResult | None) -> float | None:
    if result is None:
        return None
    if result.structural_spec_metric is not None:
        value = _maybe_float(getattr(result.structural_spec_metric, "score", None))
        if value is not None:
            return value
    if result.evaluation_summary is not None:
        value = _maybe_float(getattr(result.evaluation_summary, "structural_spec_metric", None))
        if value is not None:
            return value
        scores = getattr(result.evaluation_summary, "benchmark_scores", {}) or {}
        for key in ("spec_score", "structural_spec_metric"):
            value = _maybe_float(scores.get(key))
            if value is not None:
                return value
    return None


def _generated_spec_path(result: PipelineResult | None) -> str | None:
    if result is None:
        return None
    paths = result.artifact_paths or {}
    for key in ("vega_spec", "spec_validation"):
        value = paths.get(key)
        if value:
            return str(value)
    if result.vega_spec is not None:
        artifacts = getattr(result.vega_spec, "generation_artifacts", {}) or {}
        for value in artifacts.values():
            if value and str(value).endswith(".json"):
                return str(value)
    return None


def _valid_spec(result: PipelineResult | None) -> bool | None:
    if result is None or result.spec_validation is None:
        return None
    return bool(result.spec_validation.is_valid)


def _empty_chart(result: PipelineResult | None) -> bool | None:
    if result is None or result.empty_chart_check is None:
        return None
    check = result.empty_chart_check
    return bool(check.empty_chart_signal or check.empty_chart_status == "empty")


def _token_usage(result: PipelineResult | None, runtime: RuntimeContext) -> dict[str, int]:
    usage = result.token_usage_summary if result is not None else runtime.token_usage_summary()
    return {
        "prompt_tokens": int(usage.prompt_tokens),
        "completion_tokens": int(usage.completion_tokens),
        "total_tokens": int(usage.total_tokens),
    }


def _effective_run_status(result: PipelineResult | None, requested_status: str) -> str:
    normalized = str(requested_status or "").lower()
    if normalized in {"failed", "error"}:
        return "error"
    if result is None:
        return "error"
    if _valid_spec(result) is False or _empty_chart(result) is True:
        return "error"
    if not _path_exists(_path_from_plot_image(result.plot_image)):
        return "error"
    semantic_status = (_semantic_status(result) or "").lower()
    quality_status = (_chart_quality_status(result) or "").lower()
    issue_counts = _chart_quality_issue_counts(result)
    judge = result.visual_chart_judge
    answer_judge = result.chart_answer_judge
    if semantic_status == "failed":
        return "partial"
    if quality_status == "fail" or issue_counts["critical"] >= 2:
        return "partial"
    if quality_status == "retry" or issue_counts["critical"] == 1:
        return "partial"
    if judge is not None:
        recommendation = str(getattr(judge, "retry_recommendation", "") or "").lower()
        if bool(getattr(judge, "is_blank_or_unreadable", False)):
            return "error"
        if not bool(getattr(judge, "answers_user_query", False)) or recommendation in {"retry", "reject"}:
            return "partial"
    if answer_judge is not None:
        recommendation = str(getattr(answer_judge, "retry_recommendation", "") or "").lower()
        if not bool(getattr(answer_judge, "answers_user_query", False)) or recommendation in {"retry", "reject"}:
            return "partial"
    return "completed"


def build_task_request_payload(request: PipelineRequest) -> dict[str, Any]:
    return {
        "run_id": request.run_id,
        "query": request.query,
        "data_path": request.data_path,
        "user_context": request.user_context or {},
        "created_at": _now_iso(),
    }


def build_run_report(
        *,
        request: PipelineRequest,
        result: PipelineResult | None,
        runtime: RuntimeContext,
        status: str,
        error_type: str | None = None,
        error: str | None = None,
) -> dict[str, Any]:
    generated_image_path = _path_from_plot_image(result.plot_image if result is not None else None)
    generated_spec_path = _generated_spec_path(result)
    tokens = _token_usage(result, runtime)
    duration_seconds = _stage_duration_seconds(result)

    publication_scores = _visual_publication_scores(result)

    effective_status = _effective_run_status(result, status)
    quality_counts = _chart_quality_issue_counts(result)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": _now_iso(),
        "run_id": request.run_id,
        "query": request.query,
        "data_path": request.data_path,
        "status": effective_status,
        "pipeline_status": status,
        "semantic_status": _semantic_status(result),
        "error_type": error_type,
        "error": error,
        "has_plot": bool(generated_image_path),
        "generated_image_path": generated_image_path,
        "generated_image_exists": _path_exists(generated_image_path),
        "generated_spec_path": generated_spec_path,
        "generated_spec_exists": _path_exists(generated_spec_path),
        "valid_spec": _valid_spec(result),
        "render_success": bool(generated_image_path),
        "empty_chart": _empty_chart(result),
        "spec_score": _spec_score(result),
        "vision_score": _vision_score(result),
        "chart_quality_status": _chart_quality_status(result),
        "chart_quality_score": _chart_quality_score(result),
        "chart_quality_critical_count": quality_counts["critical"],
        "chart_quality_warning_count": quality_counts["warning"],
        **publication_scores,
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "duration_seconds": duration_seconds,
        "stage_count": len(result.stage_execution_logs) if result is not None else len(runtime.stage_execution_logs),
        "model_call_count": len(result.model_call_logs) if result is not None else len(runtime.model_call_logs),
        "artifact_paths": result.artifact_paths if result is not None else {},
    }
    report["metrics"] = {
        "valid_spec": report["valid_spec"],
        "render_success": report["render_success"],
        "empty_chart": report["empty_chart"],
        "spec_score": report["spec_score"],
        "vision_score": report["vision_score"],
        "chart_quality_status": report["chart_quality_status"],
        "chart_quality_score": report["chart_quality_score"],
        "chart_quality_critical_count": report["chart_quality_critical_count"],
        "chart_quality_warning_count": report["chart_quality_warning_count"],
        "plot_area_usage_score": report["plot_area_usage_score"],
        "axis_domain_score": report["axis_domain_score"],
        "layout_compactness_score": report["layout_compactness_score"],
        "repeat_axis_label_score": report["repeat_axis_label_score"],
        "publication_layout_score": report["publication_layout_score"],
        "prompt_tokens": report["prompt_tokens"],
        "completion_tokens": report["completion_tokens"],
        "total_tokens": report["total_tokens"],
        "duration_seconds": report["duration_seconds"],
    }
    return report


def save_task_request(runtime: RuntimeContext, request: PipelineRequest) -> str:
    return runtime.save_json_artifact(
        "task_request.json",
        build_task_request_payload(request),
        run_id=request.run_id,
    )


def save_run_report(
        runtime: RuntimeContext,
        *,
        request: PipelineRequest,
        result: PipelineResult | None,
        status: str,
        error_type: str | None = None,
        error: str | None = None,
) -> str:
    return runtime.save_json_artifact(
        "run_report.json",
        build_run_report(
            request=request,
            result=result,
            runtime=runtime,
            status=status,
            error_type=error_type,
            error=error,
        ),
        run_id=request.run_id,
    )


def save_error_report(
        runtime: RuntimeContext,
        *,
        request: PipelineRequest,
        error_type: str,
        error: str,
        traceback_text: str | None = None,
) -> str:
    payload = {
        "run_id": request.run_id,
        "query": request.query,
        "data_path": request.data_path,
        "error_type": error_type,
        "error": error,
        "traceback": traceback_text,
        "created_at": _now_iso(),
    }
    return runtime.save_json_artifact("errors.json", payload, run_id=request.run_id)
