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
        rubric_json = analysis_rubric.model_dump_json()
        if len(rubric_json) > 1800:
            rubric_json = rubric_json[:1776].rstrip() + "\n[truncated]"
        return (
            "Role: expert scientific chart analyst. Analyze only the attached accepted chart image. "
            "You cannot see the source table, hidden data, or tooltips. Do not request regeneration.\n\n"
            f"Rubric JSON:\n{rubric_json}\n\n"
            "Return structured JSON: summary, key_findings, caveats, suggested_followup_questions, "
            "visual_observations, extracted_visual_facts, confidence. Ground every finding in visible chart evidence. "
            "If labels, legends, or axes are unreadable, state that as a caveat."
        )
