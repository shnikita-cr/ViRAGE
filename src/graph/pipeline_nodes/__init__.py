from __future__ import annotations

from src.graph.pipeline_nodes.analysis_nodes import AnalysisPipelineNodesMixin
from src.graph.pipeline_nodes.common import BasePipelineNodes
from src.graph.pipeline_nodes.data_nodes import DataPipelineNodesMixin
from src.graph.pipeline_nodes.generation_nodes import GenerationPipelineNodesMixin
from src.graph.pipeline_nodes.validation_nodes import ValidationPipelineNodesMixin
from src.graph.pipeline_nodes.visual_feedback_nodes import VisualFeedbackPipelineNodesMixin


class PipelineNodes(
    DataPipelineNodesMixin,
    GenerationPipelineNodesMixin,
    ValidationPipelineNodesMixin,
    VisualFeedbackPipelineNodesMixin,
    AnalysisPipelineNodesMixin,
    BasePipelineNodes,
):
    pass
