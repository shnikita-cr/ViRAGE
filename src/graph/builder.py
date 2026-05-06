from __future__ import annotations

from typing import Any

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.graph.nodes import PipelineNodes
from src.infrastructure.runtime import RuntimeContext


def _mark_completed(state: PipelineState) -> dict[str, Any]:
    return {"stage": PipelineStage.COMPLETED}


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
        graph.add_node(name, callable_node)

    graph.add_edge(START, ordered_steps[0][0])
    for (source, _), (target, _) in zip(ordered_steps, ordered_steps[1:]):
        graph.add_edge(source, target)
    graph.add_edge(ordered_steps[-1][0], END)
    return graph.compile()
