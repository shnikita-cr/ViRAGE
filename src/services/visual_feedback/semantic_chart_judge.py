from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from src.domain.models import (
    ChartAnswerJudgeResult,
    ChartFactSummaryResult,
    ChartGroundedAnalysisRecord,
    ChartRevisionRecord,
    DataProfile,
    PlotImageArtifact,
    RequestAnalysisResult,
    SemanticChartJudgeResult,
    VegaLiteSpecArtifact,
    VLMChartDescriptionResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured_multimodal
from src.services.base import BaseService
from src.services.visual_feedback.chart_answer_judge import _normalize_retry_recommendation


class _SemanticChartJudgeSchema(BaseModel):
    chart_description: str = ""
    detected_chart_type: str | None = None
    visible_axes: dict[str, str] = Field(default_factory=dict)
    visible_legend: dict[str, Any] = Field(default_factory=dict)
    visible_labels: list[str] = Field(default_factory=list)
    visible_fields: list[str] = Field(default_factory=list)
    observed_facts: list[str] = Field(default_factory=list)
    answers_user_query: bool = False
    supports_visible_claims: bool = True
    confidence: float = 0.0
    retry_recommendation: str = "retry"
    missing_requirements: list[str] = Field(default_factory=list)
    wrong_or_suspicious_parts: list[str] = Field(default_factory=list)
    readability_issues: list[str] = Field(default_factory=list)
    improvement_comments: list[str] = Field(default_factory=list)
    feedback_for_next_generation: str = ""
    is_blank_or_unreadable: bool = False
    rationales: dict[str, str] = Field(default_factory=dict)


class SemanticChartJudgeService(BaseService):
    """Strict one-call VLM semantic judge for the rendered chart.

    The old debug chain describes the image, summarizes facts, and then judges with a text LLM. This service keeps
    the same information in one structured multimodal call, so retry decisions are grounded in the actual PNG and the
    request/spec context simultaneously.
    """

    def invoke(
        self,
        *,
        query: str,
        plot_image: PlotImageArtifact,
        vega_spec: VegaLiteSpecArtifact,
        runtime: RuntimeContext,
        request_analysis: RequestAnalysisResult | None = None,
        data_profile: DataProfile | None = None,
        compact_data_profile: dict[str, Any] | None = None,
    ) -> SemanticChartJudgeResult:
        if runtime.vlm is None:
            raise RuntimeError("Semantic chart judge requires runtime.vlm. No multimodal model was provided.")
        prompt = self._prompt(
            query=query,
            vega_spec=vega_spec,
            request_analysis=request_analysis,
            data_profile=data_profile,
            compact_data_profile=compact_data_profile,
        )
        parsed = invoke_structured_multimodal(
            runtime.vlm,
            prompt,
            plot_image.image_path,
            _SemanticChartJudgeSchema,
            runtime=runtime,
            stage="semantic_chart_judge",
            role="vlm",
            examples=[self._example()],
            max_attempts=2,
        )
        missing = _clean_list(parsed.missing_requirements)
        wrong = _clean_list(parsed.wrong_or_suspicious_parts)
        readability = _clean_list(parsed.readability_issues)
        comments = _clean_list(parsed.improvement_comments)
        feedback = str(parsed.feedback_for_next_generation or "").strip()
        recommendation = _normalize_retry_recommendation(
            parsed.retry_recommendation,
            answers_user_query=bool(parsed.answers_user_query),
            feedback_for_next_generation=feedback,
            missing_requirements=missing,
            wrong_or_suspicious_parts=[*wrong, *readability],
            improvement_comments=comments,
        )
        if parsed.is_blank_or_unreadable:
            recommendation = "retry"
            if "The rendered chart appears blank or unreadable." not in missing:
                missing.append("The rendered chart appears blank or unreadable.")
        return SemanticChartJudgeResult(
            chart_description=str(parsed.chart_description or "").strip(),
            detected_chart_type=parsed.detected_chart_type,
            visible_axes=dict(parsed.visible_axes or {}),
            visible_legend=dict(parsed.visible_legend or {}),
            visible_labels=_clean_list(parsed.visible_labels),
            visible_fields=_clean_list(parsed.visible_fields),
            observed_facts=_clean_list(parsed.observed_facts),
            answers_user_query=bool(parsed.answers_user_query),
            supports_visible_claims=bool(parsed.supports_visible_claims),
            confidence=max(0.0, min(1.0, float(parsed.confidence or 0.0))),
            retry_recommendation=recommendation,
            missing_requirements=missing,
            wrong_or_suspicious_parts=wrong,
            readability_issues=readability,
            improvement_comments=comments,
            feedback_for_next_generation=feedback,
            is_blank_or_unreadable=bool(parsed.is_blank_or_unreadable),
            rationales=dict(parsed.rationales or {}),
        )

    @staticmethod
    def _prompt(
        *,
        query: str,
        vega_spec: VegaLiteSpecArtifact,
        request_analysis: RequestAnalysisResult | None,
        data_profile: DataProfile | None,
        compact_data_profile: dict[str, Any] | None,
    ) -> str:
        spec = getattr(vega_spec, "spec_without_runtime_data", None) or vega_spec.spec_json
        context = {
            "user_query": query,
            "request_analysis": request_analysis.model_dump() if request_analysis is not None else None,
            "compact_data_profile": compact_data_profile,
            "data_profile_summary": None if data_profile is None else {
                "row_count": data_profile.row_count,
                "column_count": data_profile.col_count,
                "field_roles": data_profile.field_roles,
            },
            "generated_vega_lite_spec_without_runtime_data": _strip_large_data(spec),
        }
        return (
            "You are SemanticChartJudgeAI. Judge the attached rendered chart image strictly.\n"
            "Use the PNG as the source of truth for what is visually shown. Use the user request, request analysis, "
            "data profile, and Vega-Lite spec only to check whether the image answers the request and whether field "
            "usage is grounded.\n\n"
            "Return concrete feedback only when a retry is needed. Do not request retry without actionable feedback.\n"
            "Set retry_recommendation='accept' when the chart answers the request and there are no material issues.\n"
            "Set retry_recommendation='retry' only when missing_requirements, readability_issues, "
            "wrong_or_suspicious_parts, or feedback_for_next_generation are non-empty.\n"
            "Set retry_recommendation='reject' only if the image is unusable and cannot be repaired by a better spec.\n"
            "Strict chart quality requirements: axis titles must be readable and must name source fields and aggregation; "
            "legends are mandatory whenever color/shape/size/strokeDash or multiple metric series are used; category labels "
            "must fit without overlap or cropping; multi-metric charts must make each metric name clear and avoid misleading "
            "shared scales; vague titles such as value/total/count are not acceptable unless the field and aggregation are clear. "
            "If the chart contains both an overloaded legend and labels that do not fit, do not accept it just because both exist; "
            "require a cleaner alternative such as facet/repeat panels, horizontal bars, shorter axis titles, direct labels/tooltips, "
            "or independent scales. Prefer the clearest readable representation over preserving every visual element. "
            "Check label readability, axis/legend titles, visible fields, transformations, and whether the chart supports "
            "the requested comparison/distribution/trend/correlation. If any of these requirements fail, return retry with "
            "specific feedback for next generation.\n\n"
            f"Context JSON:\n{json.dumps(context, ensure_ascii=False, indent=2, default=str)}\n"
        )

    @staticmethod
    def _example() -> dict[str, Any]:
        return {
            "chart_description": "The image shows a bar chart comparing average PSNR by method.",
            "detected_chart_type": "bar",
            "visible_axes": {"x": "Method", "y": "mean PSNR"},
            "visible_legend": {},
            "visible_labels": ["Method", "mean PSNR"],
            "visible_fields": ["Method", "PSNR"],
            "observed_facts": ["Bars compare methods by a quantitative metric."],
            "answers_user_query": True,
            "supports_visible_claims": True,
            "confidence": 0.9,
            "retry_recommendation": "accept",
            "missing_requirements": [],
            "wrong_or_suspicious_parts": [],
            "readability_issues": [],
            "improvement_comments": [],
            "feedback_for_next_generation": "",
            "is_blank_or_unreadable": False,
            "rationales": {"prompt_compliance": "The visible chart contains the requested fields."},
        }


class SemanticChartJudgeAdapters:
    @staticmethod
    def to_vlm_description(result: SemanticChartJudgeResult) -> VLMChartDescriptionResult:
        return VLMChartDescriptionResult(
            input_scope="png_only",
            visual_description=result.chart_description,
            detected_chart_type=result.detected_chart_type,
            visible_axes=result.visible_axes,
            visible_legend=result.visible_legend,
            visible_labels=result.visible_labels,
            visible_trends=[],
            visible_comparisons=result.observed_facts,
            visible_outliers=[],
            readability_issues=result.readability_issues,
            uncertainties=[] if result.confidence >= 0.7 else ["semantic_chart_judge_low_confidence"],
            confidence=result.confidence,
        )

    @staticmethod
    def to_fact_summary(result: SemanticChartJudgeResult) -> ChartFactSummaryResult:
        return ChartFactSummaryResult(
            chart_type=result.detected_chart_type,
            facts=result.observed_facts,
            axes=result.visible_axes,
            legend=result.visible_legend,
            visible_variables=result.visible_fields,
            visible_relationships=result.observed_facts,
            uncertainties=[] if result.confidence >= 0.7 else ["semantic_chart_judge_low_confidence"],
            quality_notes=[*result.readability_issues, *result.wrong_or_suspicious_parts],
        )

    @staticmethod
    def to_answer_judge(result: SemanticChartJudgeResult) -> ChartAnswerJudgeResult:
        return ChartAnswerJudgeResult(
            answers_user_query=result.answers_user_query,
            confidence=result.confidence,
            retry_recommendation=result.retry_recommendation,
            missing_requirements=result.missing_requirements,
            wrong_or_suspicious_parts=[*result.wrong_or_suspicious_parts, *result.readability_issues],
            improvement_comments=result.improvement_comments,
            feedback_for_next_generation=result.feedback_for_next_generation,
        )

    @staticmethod
    def to_chart_analysis_record(
        *,
        query: str,
        result: SemanticChartJudgeResult,
        used_fields: list[str],
    ) -> ChartGroundedAnalysisRecord:
        return ChartGroundedAnalysisRecord(
            user_query=query,
            used_fields=used_fields,
            chart_type=result.detected_chart_type,
            observed_facts=result.observed_facts,
            issues=[*result.missing_requirements, *result.wrong_or_suspicious_parts, *result.readability_issues],
            feedback_for_next_generation=result.feedback_for_next_generation,
            accepted=result.retry_recommendation == "accept" and result.answers_user_query,
            retry_recommendation=result.retry_recommendation,
            confidence=result.confidence,
        )

    @staticmethod
    def to_revision_record(
        *,
        attempt_number: int,
        query: str,
        result: SemanticChartJudgeResult,
        vega_spec: VegaLiteSpecArtifact,
        rendered_png_path: str,
        retry_reasons: list[str],
    ) -> ChartRevisionRecord:
        return ChartRevisionRecord(
            attempt_number=attempt_number,
            user_query=query,
            accepted=result.retry_recommendation == "accept" and result.answers_user_query,
            retry_recommendation=result.retry_recommendation,
            retry_reasons=retry_reasons,
            feedback_for_next_generation=result.feedback_for_next_generation,
            generated_spec=vega_spec.spec_json,
            rendered_png_path=rendered_png_path,
        )


def _clean_list(values: list[str]) -> list[str]:
    return [str(item).strip() for item in values if str(item).strip()]


def _strip_large_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_large_data(item) for key, item in value.items() if key not in {"data", "datasets"}}
    if isinstance(value, list):
        return [_strip_large_data(item) for item in value]
    return value
