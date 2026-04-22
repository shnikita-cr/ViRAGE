from typing import Any

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.graph.nodes import PipelineNodes
from src.infrastructure.runtime import RuntimeContext


class SequentialCompiledGraph:
    def __init__(self, nodes: PipelineNodes) -> None:
        self.nodes = nodes

    def invoke(self, state: PipelineState, config: dict[str, Any] | None = None) -> PipelineState:
        current = dict(state)
        for fn in [
            self.nodes.query_understanding_node,
            self.nodes.data_profiler_node,
            self.nodes.request_analyzer_node,
            self.nodes.data_preparation_node,
            self.nodes.visrag_node,
            self.nodes.planning_node,
            self.nodes.chart_generator_node,
            self.nodes.spec_validator_node,
            self.nodes.vegalite_plot_drawing_node,
            self.nodes.scenegraph_check_node,
            self.nodes.empty_chart_check_node,
            self.nodes.spec_score_node,
            self.nodes.vlm_analysis_node,
            self.nodes.fact_extractor_node,
            self.nodes.reasoner_node,
            self.nodes.verifier_node,
            self.nodes.insights_node,
            self.nodes.vision_score_node,
            self.nodes.evaluation_summary_node,
        ]:
            current.update(fn(current))
        current["stage"] = PipelineStage.COMPLETED
        return current


def build_pipeline_graph(runtime: RuntimeContext) -> Any:
    nodes = PipelineNodes(runtime)
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return SequentialCompiledGraph(nodes)

    graph = StateGraph(PipelineState)
    graph.add_node("query_understanding", nodes.query_understanding_node)
    graph.add_node("data_profiler", nodes.data_profiler_node)
    graph.add_node("request_analyzer", nodes.request_analyzer_node)
    graph.add_node("data_preparation", nodes.data_preparation_node)
    graph.add_node("visrag", nodes.visrag_node)
    graph.add_node("planning", nodes.planning_node)
    graph.add_node("chart_generator", nodes.chart_generator_node)
    graph.add_node("spec_validator", nodes.spec_validator_node)
    graph.add_node("vegalite_plot_drawing", nodes.vegalite_plot_drawing_node)
    graph.add_node("scenegraph_check", nodes.scenegraph_check_node)
    graph.add_node("empty_chart_check", nodes.empty_chart_check_node)
    graph.add_node("vlm_analysis", nodes.vlm_analysis_node)
    graph.add_node("fact_extractor", nodes.fact_extractor_node)
    graph.add_node("reasoner", nodes.reasoner_node)
    graph.add_node("verifier", nodes.verifier_node)
    graph.add_node("insights", nodes.insights_node)
    graph.add_node("spec_score", nodes.spec_score_node)
    graph.add_node("vision_score", nodes.vision_score_node)
    graph.add_node("evaluation_summary", nodes.evaluation_summary_node)

    graph.add_edge(START, "query_understanding")
    graph.add_edge("query_understanding", "data_profiler")
    graph.add_edge("data_profiler", "request_analyzer")
    graph.add_edge("request_analyzer", "data_preparation")
    graph.add_edge("data_preparation", "visrag")
    graph.add_edge("visrag", "planning")
    graph.add_edge("planning", "chart_generator")
    graph.add_edge("chart_generator", "spec_validator")
    graph.add_edge("spec_validator", "vegalite_plot_drawing")
    graph.add_edge("vegalite_plot_drawing", "scenegraph_check")
    graph.add_edge("scenegraph_check", "empty_chart_check")
    graph.add_edge("empty_chart_check", "spec_score")
    graph.add_edge("spec_score", "vlm_analysis")
    graph.add_edge("vlm_analysis", "fact_extractor")
    graph.add_edge("fact_extractor", "reasoner")
    graph.add_edge("reasoner", "verifier")
    graph.add_edge("verifier", "insights")
    graph.add_edge("insights", "vision_score")
    graph.add_edge("vision_score", "evaluation_summary")
    graph.add_edge("evaluation_summary", END)
    return graph.compile()
