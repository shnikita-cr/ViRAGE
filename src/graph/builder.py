from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.domain.models import StageExecutionLog, StepLog, TokenUsage
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




def _enrich_step_logs_with_duration(
    *,
    state: PipelineState,
    output: dict[str, Any] | None,
    duration_ms: float,
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
        details.setdefault("duration_ms", round(duration_ms, 6))
        details.setdefault("duration_seconds", round(duration_seconds, 6))
        if hasattr(log, "model_copy"):
            enriched[index] = log.model_copy(update={
                "duration_ms": round(duration_ms, 6),
                "duration_seconds": round(duration_seconds, 6),
                "details": details,
            })
        else:
            enriched[index] = log
        try:
            # Emit the enriched log as the final completed-step update. The node may have already emitted
            # an immediate start/update log; UI collapse keeps this latest one with duration.
            from src.domain.models import StepLog

            if isinstance(enriched[index], StepLog):
                # Runtime is not available here; caller emits after this helper.
                pass
        except Exception:
            pass
    output["step_logs"] = enriched

_STAGE_TITLES = {
    "data_profiler": "Data profiling",
    "query_understanding": "Query understanding",
    "request_analyzer": "Request analysis",
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
    "vlm_chart_description": "PNG-only VLM description",
    "chart_fact_summary": "Chart fact summary",
    "chart_answer_judge": "Semantic answer judge",
    "semantic_decision": "Semantic retry decision",
    "feedback_corpus_writer": "Feedback corpus writer",
    "vlm_analysis": "VLM analysis",
    "fact_extractor": "Visual fact extraction",
    "reasoner": "Insight reasoning",
    "insights": "Insight formatting",
    "vision_score": "Vision score",
    "evaluation_summary": "Evaluation summary",
    "completed": "Completed",
}


def _emit_stage_started(runtime: RuntimeContext, name: str) -> None:
    runtime.emit_step(
        StepLog(
            stage=name,
            title=_STAGE_TITLES.get(name, name.replace("_", " ").title()),
            summary="Running now",
            details={"status": "running"},
        )
    )


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
            _emit_stage_started(runtime, name)
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
            if isinstance(output, dict):
                _enrich_step_logs_with_duration(
                    state=state,
                    output=output,
                    duration_ms=log.duration_ms,
                    duration_seconds=log.duration_seconds,
                )
                previous_count = len(state.get("step_logs", []))
                step_logs = output.get("step_logs")
                if isinstance(step_logs, list) and len(step_logs) > previous_count:
                    for item in step_logs[previous_count:]:
                        runtime.emit_step(item)
            if status == "succeeded" and isinstance(output, dict):
                output["stage_execution_logs"] = [*state.get("stage_execution_logs", []), enriched_log]

    return wrapped



def _route_technical_decision(state: PipelineState) -> str:
    status = str(state.get("technical_status") or "")
    if status == "retry":
        return "retry"
    return "ok"


def _route_semantic_gate(state: PipelineState) -> str:
    status = str(state.get("semantic_status") or "")
    if status == "enabled":
        return "enabled"
    return "disabled"


def _route_semantic_decision(state: PipelineState) -> str:
    status = str(state.get("semantic_status") or "")
    if status == "retry":
        return "retry"
    return "done"


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

    base_steps = [
        ("data_profiler", nodes.data_profiler_node),
        ("query_understanding", nodes.query_understanding_node),
        ("request_analyzer", nodes.request_analyzer_node),
        ("data_preparation", nodes.data_preparation_node),
        ("visrag", nodes.visrag_node),
        ("chart_generator", nodes.chart_generator_node),
        ("spec_validator", nodes.spec_validator_node),
        ("technical_decision", nodes.technical_decision_node),
        ("vegalite_plot_drawing", nodes.vegalite_plot_drawing_node),
        ("scenegraph_check", nodes.scenegraph_check_node),
        ("empty_chart_check", nodes.empty_chart_check_node),
        ("spec_score", nodes.spec_score_node),
        ("semantic_loop_gate", nodes.semantic_loop_gate_node),
        ("vlm_chart_description", nodes.vlm_chart_description_node),
        ("chart_fact_summary", nodes.chart_fact_summary_node),
        ("chart_answer_judge", nodes.chart_answer_judge_node),
        ("semantic_decision", nodes.semantic_decision_node),
        ("feedback_corpus_writer", nodes.feedback_corpus_writer_node),
        ("vlm_analysis", nodes.vlm_analysis_node),
        ("fact_extractor", nodes.fact_extractor_node),
        ("reasoner", nodes.reasoner_node),
        ("insights", nodes.insights_node),
        ("vision_score", nodes.vision_score_node),
        ("evaluation_summary", nodes.evaluation_summary_node),
        ("completed", _mark_completed),
    ]

    for name, callable_node in base_steps:
        graph.add_node(name, _wrap_stage_node(name=name, callable_node=callable_node, runtime=runtime))

    graph.add_edge(START, "data_profiler")
    graph.add_edge("data_profiler", "query_understanding")
    graph.add_edge("query_understanding", "request_analyzer")
    graph.add_edge("request_analyzer", "data_preparation")
    graph.add_edge("data_preparation", "visrag")
    graph.add_edge("visrag", "chart_generator")
    graph.add_edge("chart_generator", "spec_validator")
    graph.add_edge("spec_validator", "technical_decision")
    graph.add_conditional_edges(
        "technical_decision",
        _route_technical_decision,
        {"retry": "chart_generator", "ok": "vegalite_plot_drawing"},
    )

    graph.add_edge("vegalite_plot_drawing", "scenegraph_check")
    graph.add_edge("scenegraph_check", "empty_chart_check")
    graph.add_edge("empty_chart_check", "spec_score")
    graph.add_edge("spec_score", "semantic_loop_gate")
    graph.add_conditional_edges(
        "semantic_loop_gate",
        _route_semantic_gate,
        {"enabled": "vlm_chart_description", "disabled": "vlm_analysis"},
    )
    graph.add_edge("vlm_chart_description", "chart_fact_summary")
    graph.add_edge("chart_fact_summary", "chart_answer_judge")
    graph.add_edge("chart_answer_judge", "semantic_decision")
    graph.add_conditional_edges(
        "semantic_decision",
        _route_semantic_decision,
        {"retry": "feedback_corpus_writer", "done": "vlm_analysis"},
    )
    graph.add_edge("feedback_corpus_writer", "chart_generator")

    graph.add_edge("vlm_analysis", "fact_extractor")
    graph.add_edge("fact_extractor", "reasoner")
    graph.add_edge("reasoner", "insights")
    graph.add_edge("insights", "vision_score")
    graph.add_edge("vision_score", "evaluation_summary")
    graph.add_edge("evaluation_summary", "completed")
    graph.add_edge("completed", END)
    return graph.compile()
