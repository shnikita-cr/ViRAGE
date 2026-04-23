from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import PlotImageArtifact, QueryUnderstandingResult, VisualQualityMetric
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import ainvoke_structured_multimodal, invoke_structured_multimodal
from src.services.base import BaseService


class _VisionScoreSchema(BaseModel):
    visualization_type: int = Field(default=0, ge=0, le=2)
    data_encoding: int = Field(default=0, ge=0, le=2)
    data_transformation: int = Field(default=0, ge=0, le=2)
    aesthetics: int = Field(default=0, ge=0, le=2)
    prompt_compliance: int = Field(default=0, ge=0, le=2)
    insight_supportiveness: int = Field(default=0, ge=0, le=2)
    is_blank: bool = False
    details: list[str] = Field(default_factory=list)


class VisionScoreService(BaseService):
    _WEIGHTS = {
        'visualization_type': 1.0,
        'data_encoding': 2.0,
        'data_transformation': 1.0,
        'aesthetics': 0.75,
        'prompt_compliance': 1.5,
        'insight_supportiveness': 1.0,
    }

    def invoke(self, plot_image: PlotImageArtifact, runtime: RuntimeContext, query_understanding: QueryUnderstandingResult | None = None) -> VisualQualityMetric:
        if runtime.vision_judge_llm is None:
            raise RuntimeError('Vision scoring requires runtime.vision_judge_llm. No vision-judge model was provided.')
        prompt = self._build_prompt(query_understanding)
        parsed = invoke_structured_multimodal(runtime.vision_judge_llm, prompt, plot_image.image_path, _VisionScoreSchema, runtime=runtime, stage='vision_score', role='vision_judge', examples=[{'visualization_type': 2, 'data_encoding': 2, 'data_transformation': 1, 'aesthetics': 2, 'prompt_compliance': 2, 'insight_supportiveness': 2, 'is_blank': False, 'details': ['clear trend chart']}], max_attempts=2)
        return self._to_metric(parsed)

    async def ainvoke(self, plot_image: PlotImageArtifact, runtime: RuntimeContext, query_understanding: QueryUnderstandingResult | None = None) -> VisualQualityMetric:
        if runtime.vision_judge_llm is None:
            raise RuntimeError('Vision scoring requires runtime.vision_judge_llm. No vision-judge model was provided.')
        prompt = self._build_prompt(query_understanding)
        parsed = await ainvoke_structured_multimodal(runtime.vision_judge_llm, prompt, plot_image.image_path, _VisionScoreSchema, runtime=runtime, stage='vision_score', role='vision_judge', examples=[{'visualization_type': 2, 'data_encoding': 2, 'data_transformation': 1, 'aesthetics': 2, 'prompt_compliance': 2, 'insight_supportiveness': 2, 'is_blank': False, 'details': ['clear trend chart']}], max_attempts=2)
        return self._to_metric(parsed)

    @staticmethod
    def _build_prompt(query_understanding: QueryUnderstandingResult | None) -> str:
        return (
            'Judge the chart image using criteria on a discrete 0,1,2 scale.\n'
            f"User intent: {query_understanding.intent if query_understanding else 'unknown'}\n"
            f"Analysis goal: {query_understanding.analysis_goal if query_understanding else 'unknown'}\n"
            '- visualization_type\n- data_encoding\n- data_transformation\n- aesthetics\n- prompt_compliance\n- insight_supportiveness\n'
            'Also return is_blank=true if the image is effectively blank or useless.'
        )

    def _to_metric(self, parsed: _VisionScoreSchema) -> VisualQualityMetric:
        if parsed.is_blank:
            return VisualQualityMetric(score=0.0, prompt_compliance=0.0, readability=0.0, insight_supportiveness=0.0, details=[*parsed.details, 'blank chart penalty'])
        weighted_sum = 0.0
        max_sum = 0.0
        details = list(parsed.details)
        for name, weight in self._WEIGHTS.items():
            value = getattr(parsed, name)
            weighted_sum += (float(value) / 2.0) * weight
            max_sum += weight
            details.append(f'{name}={value}/2')
        score = weighted_sum / max_sum if max_sum else 0.0
        return VisualQualityMetric(score=round(max(0.0, min(1.0, score)), 4), prompt_compliance=round(parsed.prompt_compliance / 2.0,4), readability=round(parsed.aesthetics/2.0,4), insight_supportiveness=round(parsed.insight_supportiveness/2.0,4), details=details)
