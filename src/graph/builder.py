from __future__ import annotations

from typing import Any

from src.application.runtime.state import PipelineState
from src.domain.common.enums import PipelineStage
from src.graph.nodes import PipelineNodes
from src.graph.stage_executor import wrap_stage_node
from src.infrastructure.runtime import RuntimeContext


def _mark_completed(state: PipelineState) -> dict[str, Any]:
    return {"stage": PipelineStage.COMPLETED}


def _route_technical_decision(state: PipelineState) -> str:
    status = str(state.get("technical_status") or "")
    if status == "retry":
        return "retry"
    return "ok"


def _route_semantic_gate(state: PipelineState) -> str:
    status = str(state.get("semantic_status") or "")
    if status != "enabled":
        return "disabled"
    mode = str(state.get("semantic_feedback_mode") or "strict")
    if mode == "debug_full_chain":
        return "debug_full_chain"
    return "strict"


def _route_after_spec_score(state: PipelineState, *, runtime: RuntimeContext) -> str:
    if bool(getattr(runtime.settings, "semantic_feedback_loop_enabled", True)):
        return "semantic_loop"
    if bool(getattr(runtime.settings, "analytics_tail_enabled", True)):
        return "analytics_tail"
    return "completed"


def _route_after_semantic_decision(state: PipelineState, *, runtime: RuntimeContext) -> str:
    status = str(state.get("semantic_status") or "")
    if status == "retry":
        return "retry"
    if status == "failed":
        return "failed_tail" if bool(getattr(runtime.settings, "analytics_tail_enabled", True)) else "failed_done"
    return "accepted_tail" if bool(getattr(runtime.settings, "analytics_tail_enabled", True)) else "accepted_done"


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
        ("query_request_analysis", nodes.query_request_analysis_node),
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
        ("visual_chart_judge", nodes.visual_chart_judge_node),
        ("vlm_chart_description", nodes.vlm_chart_description_node),
        ("chart_fact_summary", nodes.chart_fact_summary_node),
        ("chart_answer_judge", nodes.chart_answer_judge_node),
        ("semantic_decision", nodes.semantic_decision_node),
        ("feedback_corpus_writer", nodes.feedback_corpus_writer_node),
        ("vlm_analysis", nodes.vlm_analysis_node),
        ("evaluation_summary", nodes.evaluation_summary_node),
        ("completed", _mark_completed),
    ]

    for name, callable_node in base_steps:
        graph.add_node(name, wrap_stage_node(name=name, callable_node=callable_node, runtime=runtime))

    graph.add_edge(START, "data_profiler")
    graph.add_edge("data_profiler", "query_request_analysis")
    graph.add_edge("query_request_analysis", "data_preparation")
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
    graph.add_conditional_edges(
        "spec_score",
        lambda state: _route_after_spec_score(state, runtime=runtime),
        {"semantic_loop": "semantic_loop_gate", "analytics_tail": "vlm_analysis", "completed": "completed"},
    )
    graph.add_conditional_edges(
        "semantic_loop_gate",
        _route_semantic_gate,
        {"strict": "visual_chart_judge", "debug_full_chain": "vlm_chart_description", "disabled": "vlm_analysis"},
    )
    graph.add_edge("visual_chart_judge", "semantic_decision")
    graph.add_edge("vlm_chart_description", "chart_fact_summary")
    graph.add_edge("chart_fact_summary", "chart_answer_judge")
    graph.add_edge("chart_answer_judge", "semantic_decision")
    graph.add_conditional_edges(
        "semantic_decision",
        lambda state: _route_after_semantic_decision(state, runtime=runtime),
        {
            "retry": "feedback_corpus_writer",
            "accepted_tail": "vlm_analysis",
            "accepted_done": "completed",
            "failed_tail": "evaluation_summary",
            "failed_done": "completed",
        },
    )
    graph.add_edge("feedback_corpus_writer", "chart_generator")

    graph.add_edge("vlm_analysis", "evaluation_summary")
    graph.add_edge("evaluation_summary", "completed")
    graph.add_edge("completed", END)
    return graph.compile()
