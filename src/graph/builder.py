from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.domain.models import StageExecutionLog, TokenUsage
from src.graph.nodes import PipelineNodes
from src.infrastructure.runtime import RuntimeContext


def _mark_completed(state: PipelineState) -> dict[str, Any]:
    return {"stage": PipelineStage.COMPLETED}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_ms(started_at: datetime, finished_at: datetime) -> float:
    return max((finished_at - started_at).total_seconds() * 1000.0, 0.0)


def _state_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted(str(key) for key in value.keys())
    return []


def _pipeline_stage_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, PipelineStage):
        return value.value
    return str(value)


def _sum_token_usage(logs: list[Any]) -> TokenUsage:
    usage = TokenUsage()
    for item in logs:
        token_usage = getattr(item, "token_usage", None)
        if token_usage is None:
            continue
        usage.prompt_tokens += int(getattr(token_usage, "prompt_tokens", 0) or 0)
        usage.completion_tokens += int(getattr(token_usage, "completion_tokens", 0) or 0)
        usage.total_tokens += int(getattr(token_usage, "total_tokens", 0) or 0)
    return usage


def _artifact_paths_delta(state: PipelineState, output: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(output, dict):
        return {}
    before = state.get("artifact_paths", {})
    after = output.get("artifact_paths")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {}
    delta: dict[str, str] = {}
    for key, value in after.items():
        if before.get(key) != value:
            delta[str(key)] = str(value)
    return delta


def _wrap_stage_node(
    *,
    name: str,
    callable_node: Callable[[PipelineState], dict[str, Any]],
    runtime: RuntimeContext,
) -> Callable[[PipelineState], dict[str, Any]]:
    def wrapped(state: PipelineState) -> dict[str, Any]:
        started = datetime.now(timezone.utc)
        model_call_start_count = len(runtime.model_call_logs)
        input_keys = _state_keys(state)
        output: dict[str, Any] | None = None
        status = "succeeded"
        error: str | None = None

        try:
            output = callable_node(state)
            if output is None:
                output = {}
            if not isinstance(output, dict):
                raise TypeError(f"Pipeline node {name!r} must return a dict, got {type(output).__name__}.")
            return output
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            finished = datetime.now(timezone.utc)
            model_call_end_count = len(runtime.model_call_logs)
            stage_calls = runtime.model_call_logs[model_call_start_count:model_call_end_count]
            output_keys = _state_keys(output)
            pipeline_stage = _pipeline_stage_value(output.get("stage") if isinstance(output, dict) else None)
            artifact_paths = _artifact_paths_delta(state, output)

            log = StageExecutionLog(
                node_name=name,
                pipeline_stage=pipeline_stage or name,
                status=status,
                started_at=started.isoformat(),
                finished_at=finished.isoformat(),
                duration_ms=round(_duration_ms(started, finished), 6),
                duration_seconds=round(max((finished - started).total_seconds(), 0.0), 6),
                input_keys=input_keys,
                output_keys=output_keys,
                artifact_paths=artifact_paths,
                model_call_start_index=model_call_start_count + 1 if stage_calls else model_call_start_count,
                model_call_end_index=model_call_end_count,
                model_call_count=len(stage_calls),
                token_usage=_sum_token_usage(stage_calls),
                error=error,
            )
            enriched_log = runtime.add_stage_execution_log(log)
            if status == "succeeded" and isinstance(output, dict):
                output["stage_execution_logs"] = [*state.get("stage_execution_logs", []), enriched_log]

    return wrapped


def build_pipeline_graph(runtime: RuntimeContext):
    """Build the ViRAGE pipeline with LangGraph instead of a custom sequential runner."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is required to build the ViRAGE pipeline graph. Install project requirements first."
        ) from exc

    nodes = PipelineNodes(runtime)
    graph = StateGraph(PipelineState)

    ordered_steps = [
        ("data_profiler", nodes.data_profiler_node),
        ("query_understanding", nodes.query_understanding_node),
        ("request_analyzer", nodes.request_analyzer_node),
        ("data_preparation", nodes.data_preparation_node),
        ("visrag", nodes.visrag_node),
        ("chart_generator", nodes.chart_generator_node),
        ("spec_validator", nodes.spec_validator_node),
        ("vegalite_plot_drawing", nodes.vegalite_plot_drawing_node),
        ("scenegraph_check", nodes.scenegraph_check_node),
        ("empty_chart_check", nodes.empty_chart_check_node),
        ("spec_score", nodes.spec_score_node),
        ("vlm_analysis", nodes.vlm_analysis_node),
        ("fact_extractor", nodes.fact_extractor_node),
        ("reasoner", nodes.reasoner_node),
        ("verifier", nodes.verifier_node),
        ("insights", nodes.insights_node),
        ("vision_score", nodes.vision_score_node),
        ("evaluation_summary", nodes.evaluation_summary_node),
        ("completed", _mark_completed),
    ]

    for name, callable_node in ordered_steps:
        graph.add_node(name, _wrap_stage_node(name=name, callable_node=callable_node, runtime=runtime))

    graph.add_edge(START, ordered_steps[0][0])
    for (source, _), (target, _) in zip(ordered_steps, ordered_steps[1:]):
        graph.add_edge(source, target)
    graph.add_edge(ordered_steps[-1][0], END)
    return graph.compile()
