from typing import Any

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.graph.nodes import PipelineNodes
from src.graph.routing import route_case
from src.infrastructure.runtime import RuntimeContext


class SequentialCompiledGraph:
    def __init__(self, nodes: PipelineNodes) -> None:
        self.nodes = nodes

    def invoke(self, state: PipelineState, config: dict[str, Any] | None = None) -> PipelineState:
        current = dict(state)
        current.update(self.nodes.query_understanding_node(current))
        if route_case(current) == "planning_non_canonical":
            current.update(self.nodes.planning_non_canonical_node(current))
        else:
            current.update(self.nodes.planning_canonical_node(current))
        for fn in [
            self.nodes.data_profiler_node,
            self.nodes.data_preparation_node,
            self.nodes.visrag_node,
            self.nodes.codegen_node,
            self.nodes.coderun_node,
            self.nodes.artifact_store_node,
            self.nodes.chart_reader_node,
            self.nodes.fact_extractor_node,
            self.nodes.reasoner_node,
            self.nodes.verifier_node,
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
    graph.add_node("planning_canonical", nodes.planning_canonical_node)
    graph.add_node("planning_non_canonical", nodes.planning_non_canonical_node)
    graph.add_node("data_profiler", nodes.data_profiler_node)
    graph.add_node("data_preparation", nodes.data_preparation_node)
    graph.add_node("visrag", nodes.visrag_node)
    graph.add_node("codegen", nodes.codegen_node)
    graph.add_node("coderun", nodes.coderun_node)
    graph.add_node("artifact_store", nodes.artifact_store_node)
    graph.add_node("chart_reader", nodes.chart_reader_node)
    graph.add_node("fact_extractor", nodes.fact_extractor_node)
    graph.add_node("reasoner", nodes.reasoner_node)
    graph.add_node("verifier", nodes.verifier_node)

    graph.add_edge(START, "query_understanding")
    graph.add_conditional_edges(
        "query_understanding",
        route_case,
        {
            "planning_canonical": "planning_canonical",
            "planning_non_canonical": "planning_non_canonical",
        },
    )
    graph.add_edge("planning_canonical", "data_profiler")
    graph.add_edge("planning_non_canonical", "data_profiler")
    graph.add_edge("data_profiler", "data_preparation")
    graph.add_edge("data_preparation", "visrag")
    graph.add_edge("visrag", "codegen")
    graph.add_edge("codegen", "coderun")
    graph.add_edge("coderun", "artifact_store")
    graph.add_edge("artifact_store", "chart_reader")
    graph.add_edge("chart_reader", "fact_extractor")
    graph.add_edge("fact_extractor", "reasoner")
    graph.add_edge("reasoner", "verifier")
    graph.add_edge("verifier", END)
    return graph.compile()
