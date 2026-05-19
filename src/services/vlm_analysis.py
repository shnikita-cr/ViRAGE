from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import AnalysisRubric, PlotImageArtifact, VLMAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import ainvoke_structured_multimodal, invoke_structured_multimodal
from src.services.base import BaseService


class _VLMChartAnalysisSchema(BaseModel):
    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    suggested_followup_questions: list[str] = Field(default_factory=list)
    visual_observations: list[str] = Field(default_factory=list)
    extracted_visual_facts: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class VLMAnalysisService(BaseService):
    def invoke(self, plot_image: PlotImageArtifact, analysis_rubric: AnalysisRubric,
               runtime: RuntimeContext) -> VLMAnalysisResult:
        if runtime.vlm is None:
            raise RuntimeError("Visual chart analysis requires runtime.vlm. No multimodal analysis model was provided.")
        prompt = self._prompt(analysis_rubric)
        parsed = invoke_structured_multimodal(
            runtime.vlm,
            prompt,
            plot_image.image_path,
            _VLMChartAnalysisSchema,
            runtime=runtime,
            stage="vlm_chart_analysis",
            role="vlm",
            examples=[{
                "summary": "The chart compares average PSNR values across denoising methods.",
                "key_findings": ["The tallest bar corresponds to the method with the highest mean PSNR."],
                "caveats": ["The analysis is based only on visible chart information and not the source table."],
                "suggested_followup_questions": ["How does the ranking change for SSIM or LPIPS?"],
                "visual_observations": ["Bars compare methods by a quantitative metric."],
                "extracted_visual_facts": ["The chart supports comparison of methods by mean PSNR."],
                "confidence": 0.82,
            }],
            max_attempts=2,
        )
        return VLMAnalysisResult(**parsed.model_dump())

    async def ainvoke(self, plot_image: PlotImageArtifact, analysis_rubric: AnalysisRubric,
                      runtime: RuntimeContext) -> VLMAnalysisResult:
        if runtime.vlm is None:
            raise RuntimeError("Visual chart analysis requires runtime.vlm. No multimodal analysis model was provided.")
        prompt = self._prompt(analysis_rubric)
        parsed = await ainvoke_structured_multimodal(
            runtime.vlm,
            prompt,
            plot_image.image_path,
            _VLMChartAnalysisSchema,
            runtime=runtime,
            stage="vlm_chart_analysis",
            role="vlm",
            examples=[{
                "summary": "The chart compares average PSNR values across denoising methods.",
                "key_findings": ["The tallest bar corresponds to the method with the highest mean PSNR."],
                "caveats": ["The analysis is based only on visible chart information and not the source table."],
                "suggested_followup_questions": ["How does the ranking change for SSIM or LPIPS?"],
                "visual_observations": ["Bars compare methods by a quantitative metric."],
                "extracted_visual_facts": ["The chart supports comparison of methods by mean PSNR."],
                "confidence": 0.82,
            }],
            max_attempts=2,
        )
        return VLMAnalysisResult(**parsed.model_dump())

    @staticmethod
    def _prompt(analysis_rubric: AnalysisRubric) -> str:
        return (
            "You are VLMChartAnalysisAI. Analyze only the accepted rendered chart image. "
            "Do not decide whether the chart must be regenerated; that is handled by SemanticChartJudgeAI. "
            "Do not assume access to the source table. Extract useful, chart-grounded insights for the user.\n\n"
            f"Analysis rubric JSON:\n{analysis_rubric.model_dump_json(indent=2)}\n\n"
            "Return structured JSON with:\n"
            "- summary: one concise overview of what the chart shows;\n"
            "- key_findings: useful findings grounded in visible chart evidence;\n"
            "- caveats: limits of interpretation from the chart only;\n"
            "- suggested_followup_questions: useful next analytical questions;\n"
            "- visual_observations: neutral visible observations;\n"
            "- extracted_visual_facts: facts visible in the chart;\n"
            "- confidence: 0..1.\n"
            "If labels, legends, or axes are unreadable, mention this as a caveat, but do not request retry."
        )
