from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import AnalysisRubric, PlotImageArtifact, VLMAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured_multimodal
from src.services.base import BaseService


class _VLMAnalysisSchema(BaseModel):
    visual_observations: list[str] = Field(default_factory=list)
    extracted_visual_facts: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class VLMAnalysisService(BaseService):
    def invoke(self, plot_image: PlotImageArtifact, analysis_rubric: AnalysisRubric,
               runtime: RuntimeContext) -> VLMAnalysisResult:
        if runtime.vlm is None:
            raise RuntimeError("Visual analysis requires runtime.vlm. No multimodal analysis model was provided.")
        prompt = (
            "You analyze only the chart image. Do not assume access to the source table.\n"
            f"Analysis rubric JSON:\n{analysis_rubric.model_dump_json(indent=2)}\n\n"
            "Return concise visual observations and extracted visual facts grounded in the image only."
        )
        parsed = invoke_structured_multimodal(runtime.vlm, prompt, plot_image.image_path, _VLMAnalysisSchema)
        return VLMAnalysisResult(**parsed.model_dump())
