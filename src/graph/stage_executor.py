from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from src.application.runtime.state import PipelineState
from src.domain.common.enums import PipelineStage
from src.domain.models import StageExecutionLog, StepLog, TokenUsage
from src.infrastructure.runtime import RuntimeContext

STAGE_TITLES = {
    "data_profiler": "Data profiling",
    "query_request_analysis": "Query and request analysis",
    "data_preparation": "Data preparation",
    "visrag": "Spec retrieval",
    "chart_generator": "Chart generation",
    "spec_validator": "Spec validation",
    "technical_decision": "Technical retry decision",
    "vegalite_plot_drawing": "Vega-Lite rendering",
    "scenegraph_check": "Scenegraph check",
    "empty_chart_check": "Empty chart check",
    "spec_score": "Spec score",
    "semantic_loop_gate": "Semantic VLM gate",
    "visual_chart_judge": "PNG-only visual chart judge",
    "vlm_chart_description": "PNG-only VLM description",
    "chart_fact_summary": "Chart fact summary",
    "chart_answer_judge": "Semantic answer judge",
    "semantic_decision": "Semantic retry decision",
    "feedback_corpus_writer": "Feedback corpus writer",
    "vlm_analysis": "Chart-grounded VLM analysis",
    "evaluation_summary": "Evaluation summary",
    "completed": "Completed",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def duration_ms(started_at: datetime, finished_at: datetime) -> float:
    return max((finished_at - started_at).total_seconds() * 1000.0, 0.0)


def state_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value.keys())
    return []


def pipeline_stage_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, PipelineStage):
        return value.value
    return str(value)


def sum_token_usage(logs: list[Any]) -> TokenUsage:
    usage = TokenUsage()
    for item in logs:
        token_usage = getattr(item, "token_usage", None)
        if token_usage is None:
            continue
        usage.prompt_tokens += int(getattr(token_usage, "prompt_tokens", 0) or 0)
        usage.completion_tokens += int(getattr(token_usage, "completion_tokens", 0) or 0)
        usage.total_tokens += int(getattr(token_usage, "total_tokens", 0) or 0)
    return usage


def artifact_paths_delta(state: PipelineState, output: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(output, dict):
        return {}
    before = state.get("artifact_paths", {})
    after = output.get("artifact_paths")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {}
    return {str(key): str(value) for key, value in after.items() if before.get(key) != value}


def enrich_step_logs_with_duration(
        *,
        state: PipelineState,
        output: dict[str, Any] | None,
        duration_ms_value: float,
        duration_seconds: float,
) -> None:
    if not isinstance(output, dict):
        return
    step_logs = output.get("step_logs")
    if not isinstance(step_logs, list):
        return
    previous_count = len(state.get("step_logs", []))
    if len(step_logs) <= previous_count:
        return
    enriched = list(step_logs)
    for index in range(previous_count, len(enriched)):
        log = enriched[index]
        details = dict(getattr(log, "details", {}) or {})
        details.setdefault("duration_ms", round(duration_ms_value, 6))
        details.setdefault("duration_seconds", round(duration_seconds, 6))
        if hasattr(log, "model_copy"):
            enriched[index] = log.model_copy(update={
                "duration_ms": round(duration_ms_value, 6),
                "duration_seconds": round(duration_seconds, 6),
                "details": details,
            })
    output["step_logs"] = enriched


class StageExecutor:
    def __init__(self, *, name: str, callable_node: Callable[[PipelineState], dict[str, Any]],
                 runtime: RuntimeContext) -> None:
        self.name = name
        self.callable_node = callable_node
        self.runtime = runtime

    def __call__(self, state: PipelineState) -> dict[str, Any]:
        started = utc_now()
        model_call_start_count = len(self.runtime.model_call_logs)
        input_keys = state_keys(state)
        output: dict[str, Any] | None = None
        status = "succeeded"
        error: str | None = None

        try:
            self._emit_stage_started()
            output = self.callable_node(state)
            if output is None:
                output = {}
            if not isinstance(output, dict):
                raise TypeError(
                    f"Pipeline node {self.name!r} must return a dict, got {type(output).__name__}."
                )
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            finished = utc_now()
            self._record_stage_execution(
                state=state,
                output=output,
                status=status,
                error=error,
                started=started,
                finished=finished,
                model_call_start_count=model_call_start_count,
                input_keys=input_keys,
            )
        return output

    def _emit_stage_started(self) -> None:
        self.runtime.emit_step(
            StepLog(
                stage=self.name,
                title=STAGE_TITLES.get(self.name, self.name.replace("_", " ").title()),
                summary="Running now",
                details={"status": "running"},
            )
        )

    def _record_stage_execution(
            self,
            *,
            state: PipelineState,
            output: dict[str, Any] | None,
            status: str,
            error: str | None,
            started: datetime,
            finished: datetime,
            model_call_start_count: int,
            input_keys: list[str],
    ) -> None:
        model_call_end_count = len(self.runtime.model_call_logs)
        stage_calls = self.runtime.model_call_logs[model_call_start_count:model_call_end_count]
        stage_duration_ms = round(duration_ms(started, finished), 6)
        stage_duration_seconds = round(max((finished - started).total_seconds(), 0.0), 6)
        log = StageExecutionLog(
            node_name=self.name,
            pipeline_stage=pipeline_stage_value(output.get("stage") if isinstance(output, dict) else None) or self.name,
            status=status,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            duration_ms=stage_duration_ms,
            duration_seconds=stage_duration_seconds,
            input_keys=input_keys,
            output_keys=state_keys(output),
            artifact_paths=artifact_paths_delta(state, output),
            model_call_start_index=model_call_start_count + 1 if stage_calls else model_call_start_count,
            model_call_end_index=model_call_end_count,
            model_call_count=len(stage_calls),
            token_usage=sum_token_usage(stage_calls),
            error=error,
        )
        enriched_log = self.runtime.add_stage_execution_log(log)
        if not isinstance(output, dict):
            return
        enrich_step_logs_with_duration(
            state=state,
            output=output,
            duration_ms_value=stage_duration_ms,
            duration_seconds=stage_duration_seconds,
        )
        previous_count = len(state.get("step_logs", []))
        step_logs = output.get("step_logs")
        if isinstance(step_logs, list) and len(step_logs) > previous_count:
            for item in step_logs[previous_count:]:
                self.runtime.emit_step(item)
        if status == "succeeded":
            output["stage_execution_logs"] = [*state.get("stage_execution_logs", []), enriched_log]


def wrap_stage_node(
        *,
        name: str,
        callable_node: Callable[[PipelineState], dict[str, Any]],
        runtime: RuntimeContext,
) -> Callable[[PipelineState], dict[str, Any]]:
    return StageExecutor(name=name, callable_node=callable_node, runtime=runtime)
