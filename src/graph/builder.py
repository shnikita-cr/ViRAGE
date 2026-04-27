from __future__ import annotations

from typing import Any, Callable

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.graph.nodes import PipelineNodes
from src.infrastructure.runtime import RuntimeContext


class SequentialCompiledGraph:
    def __init__(self, nodes: PipelineNodes) -> None:
        self.nodes = nodes
        self._steps: list[Callable[[PipelineState], dict[str, Any]]] = [
            nodes.query_understanding_node,
            nodes.data_profiler_node,
            nodes.request_analyzer_node,
            nodes.data_preparation_node,
            nodes.visrag_node,
            nodes.planning_node,
            nodes.chart_generator_node,
            nodes.spec_validator_node,
            nodes.vegalite_plot_drawing_node,
            nodes.scenegraph_check_node,
            nodes.empty_chart_check_node,
            nodes.spec_score_node,
            nodes.vlm_analysis_node,
            nodes.fact_extractor_node,
            nodes.reasoner_node,
            nodes.verifier_node,
            nodes.insights_node,
            nodes.vision_score_node,
            nodes.evaluation_summary_node,
        ]

    def invoke(self, state: PipelineState, config: dict[str, Any] | None = None) -> PipelineState:
        current = dict(state)
        for step in self._steps:
            updates = step(current)
            if updates:
                current.update(updates)
        current["stage"] = PipelineStage.COMPLETED
        return current


def build_pipeline_graph(runtime: RuntimeContext) -> SequentialCompiledGraph:
    return SequentialCompiledGraph(PipelineNodes(runtime))
