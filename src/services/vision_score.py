from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import PlotImageArtifact, VisualQualityMetric
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured_multimodal
from src.services.base import BaseService


class _VisionScoreSchema(BaseModel):
    visualization_type: int = Field(default=0, ge=0, le=2)
    data_encoding: int = Field(default=0, ge=0, le=2)
    data_transformation: int = Field(default=0, ge=0, le=2)
    aesthetics: int = Field(default=0, ge=0, le=2)
    prompt_compliance: int = Field(default=0, ge=0, le=2)
    is_blank: bool = False
    details: list[str] = Field(default_factory=list)


class VisionScoreService(BaseService):
    _WEIGHTS = {
        "visualization_type": 1.0,
        "data_encoding": 2.0,
        "data_transformation": 1.0,
        "aesthetics": 0.75,
        "prompt_compliance": 1.5,
    }

    def invoke(self, plot_image: PlotImageArtifact, runtime: RuntimeContext) -> VisualQualityMetric:
        if runtime.vision_judge_llm is None:
            raise RuntimeError("Vision scoring requires runtime.vision_judge_llm. No vision-judge model was provided.")
        prompt = (
            "Judge the chart image using these criteria on a discrete 0,1,2 scale:\n"
            "- visualization_type: is the chart type visually coherent for the displayed content?\n"
            "- data_encoding: are axes/marks/encodings visually interpretable?\n"
            "- data_transformation: do aggregation/binning/grouping effects look coherent?\n"
            "- aesthetics: is the chart readable and uncluttered?\n"
            "- prompt_compliance: does the chart appear aligned with a typical analytical request?\n"
            "Also return is_blank=true if the image is effectively blank or useless.\n"
            "Return structured output only."
        )
        parsed = invoke_structured_multimodal(runtime.vision_judge_llm, prompt, plot_image.image_path,
                                              _VisionScoreSchema)
        if parsed.is_blank:
            return VisualQualityMetric(score=0.0, details=[*parsed.details, "blank chart penalty"])

        weighted_sum = 0.0
        max_sum = 0.0
        details = list(parsed.details)
        for name, weight in self._WEIGHTS.items():
            value = getattr(parsed, name)
            weighted_sum += (float(value) / 2.0) * weight
            max_sum += weight
            details.append(f"{name}={value}/2")
        score = weighted_sum / max_sum if max_sum else 0.0
        return VisualQualityMetric(score=round(max(0.0, min(1.0, score)), 4), details=details)
