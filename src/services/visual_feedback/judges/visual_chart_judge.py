from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from src.domain.models import (
    ChartAnswerJudgeResult,
    ChartFactSummaryResult,
    ChartGroundedAnalysisRecord,
    ChartRevisionRecord,
    PlotImageArtifact,
    QueryRequestAnalysisResult,
    VisualChartJudgeResult,
    VLMChartDescriptionResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured_multimodal
from src.services.base import BaseService
from src.services.visual_feedback.judges.chart_answer_judge import _normalize_retry_recommendation
from src.services.visual_feedback.adapters.chartsquared_adapter import ChartSquaredAdapter


class _VisualChartJudgeSchema(BaseModel):
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
    plot_area_usage_score: float = 0.0
    axis_domain_score: float = 0.0
    layout_compactness_score: float = 0.0
    repeat_axis_label_score: float = 0.0
    publication_layout_score: float = 0.0
    plot_area_issues: list[str] = Field(default_factory=list)
    axis_domain_issues: list[str] = Field(default_factory=list)
    layout_compactness_issues: list[str] = Field(default_factory=list)
    repeat_axis_label_issues: list[str] = Field(default_factory=list)
    publication_layout_issues: list[str] = Field(default_factory=list)
    rationales: dict[str, str] = Field(default_factory=dict)


class VisualChartJudgeService(BaseService):
    """Strict one-call VLM visual judge for the rendered chart.

    The old debug chain describes the image, summarizes facts, and then judges with a text LLM. This service keeps
    the same information in one structured multimodal call, so retry decisions are grounded in the actual PNG and the
    request/spec context simultaneously.
    """

    def invoke(
            self,
            *,
            query: str,
            plot_image: PlotImageArtifact,
            runtime: RuntimeContext,
            request_analysis: QueryRequestAnalysisResult | None = None,
            visual_judge_requirements: dict[str, Any] | None = None,
    ) -> VisualChartJudgeResult:
        if runtime.vlm is None:
            raise RuntimeError("Visual chart judge requires runtime.vlm. No multimodal model was provided.")
        prompt = self._prompt(
            query=query,
            request_analysis=request_analysis,
            visual_judge_requirements=visual_judge_requirements,
            use_chartsquared=bool(getattr(runtime.settings, "visual_judge_use_chartsquared", True)),
            chartsquared_max_eval_questions=int(getattr(runtime.settings, "chartsquared_max_eval_questions", 8)),
            prompt_max_chars=int(getattr(runtime.settings, "chartsquared_prompt_max_chars", 6000)),
            chartsquared_project_root=getattr(runtime.settings, "chartsquared_project_root", None),
        )
        parsed = invoke_structured_multimodal(
            runtime.vlm,
            prompt,
            plot_image.image_path,
            _VisualChartJudgeSchema,
            runtime=runtime,
            stage="visual_chart_judge",
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
        return VisualChartJudgeResult(
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
            plot_area_usage_score=_clamp_score(parsed.plot_area_usage_score),
            axis_domain_score=_clamp_score(parsed.axis_domain_score),
            layout_compactness_score=_clamp_score(parsed.layout_compactness_score),
            repeat_axis_label_score=_clamp_score(parsed.repeat_axis_label_score),
            publication_layout_score=_clamp_score(parsed.publication_layout_score),
            plot_area_issues=_clean_list(parsed.plot_area_issues),
            axis_domain_issues=_clean_list(parsed.axis_domain_issues),
            layout_compactness_issues=_clean_list(parsed.layout_compactness_issues),
            repeat_axis_label_issues=_clean_list(parsed.repeat_axis_label_issues),
            publication_layout_issues=_clean_list(parsed.publication_layout_issues),
            rationales=dict(parsed.rationales or {}),
        )

    @staticmethod
    def _prompt(
            *,
            query: str,
            request_analysis: QueryRequestAnalysisResult | None,
            visual_judge_requirements: dict[str, Any] | None,
            use_chartsquared: bool,
            chartsquared_max_eval_questions: int,
            prompt_max_chars: int,
            chartsquared_project_root: Any | None,
    ) -> str:
        requirements = dict(visual_judge_requirements or {})
        questions = [
            str(item).strip()
            for item in requirements.get("yes_no_questions", [])[:chartsquared_max_eval_questions]
            if str(item).strip()
        ]
        if not questions and request_analysis is not None:
            questions = [
                f"Is the requested field '{field}' visibly represented in the static chart image?"
                for field in request_analysis.selected_fields[:chartsquared_max_eval_questions]
            ]
        payload = {
            "user_query": query,
            "visual_judge_requirements": {
                "must_be_visible": requirements.get("must_be_visible", []),
                "acceptable_visual_encodings": requirements.get("acceptable_visual_encodings", {}),
                "critical_failures": requirements.get("critical_failures", []),
                "yes_no_questions": questions,
            },
        }
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        payload_limit = min(prompt_max_chars, 2400)
        if len(payload_text) > payload_limit:
            payload_text = payload_text[:payload_limit] + "\n... truncated ..."

        chartsquared_rules = ""
        if use_chartsquared:
            block = ChartSquaredAdapter.build_visual_judge_block(
                project_root=chartsquared_project_root,
                requirements=requirements,
                max_questions=chartsquared_max_eval_questions,
            )
            chartsquared_rules = block.text[:1400]
        return (
            "Role: visual chart quality evaluator. Judge only the attached PNG. "
            "Do not infer from source tables, Vega-Lite specs, hidden data, or tooltips.\n"
            f"{chartsquared_rules}\n"
            "Decision: accept only if the image visibly answers the request and no critical visual requirement fails; "
            "retry if repairable; reject only if unusable. Required axes, fields, legends, grouping/facet, trends, "
            "comparisons, distributions, and relationships must be visible and readable. Tooltip-only evidence is not acceptable.\n"
            "Score plot_area_usage_score, axis_domain_score, layout_compactness_score, repeat_axis_label_score, publication_layout_score from 0..1. "
            "List concrete *_issues and feedback_for_next_generation when retry is needed.\n\n"
            f"PNG-only judge payload:\n{payload_text}\n"
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
            "plot_area_usage_score": 0.85,
            "axis_domain_score": 0.85,
            "layout_compactness_score": 0.9,
            "repeat_axis_label_score": 1.0,
            "publication_layout_score": 0.85,
            "plot_area_issues": [],
            "axis_domain_issues": [],
            "layout_compactness_issues": [],
            "repeat_axis_label_issues": [],
            "publication_layout_issues": [],
            "rationales": {"prompt_compliance": "The visible chart contains the requested fields."},
        }


def _publication_issues(result: VisualChartJudgeResult) -> list[str]:
    return [
        *result.plot_area_issues,
        *result.axis_domain_issues,
        *result.layout_compactness_issues,
        *result.repeat_axis_label_issues,
        *result.publication_layout_issues,
    ]


class VisualChartJudgeAdapters:
    @staticmethod
    def to_vlm_description(result: VisualChartJudgeResult) -> VLMChartDescriptionResult:
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
            readability_issues=[*result.readability_issues, *_publication_issues(result)],
            uncertainties=[] if result.confidence >= 0.7 else ["visual_chart_judge_low_confidence"],
            confidence=result.confidence,
        )

    @staticmethod
    def to_fact_summary(result: VisualChartJudgeResult) -> ChartFactSummaryResult:
        return ChartFactSummaryResult(
            chart_type=result.detected_chart_type,
            facts=result.observed_facts,
            axes=result.visible_axes,
            legend=result.visible_legend,
            visible_variables=result.visible_fields,
            visible_relationships=result.observed_facts,
            uncertainties=[] if result.confidence >= 0.7 else ["visual_chart_judge_low_confidence"],
            quality_notes=[*result.readability_issues, *_publication_issues(result), *result.wrong_or_suspicious_parts],
        )

    @staticmethod
    def to_answer_judge(result: VisualChartJudgeResult) -> ChartAnswerJudgeResult:
        return ChartAnswerJudgeResult(
            answers_user_query=result.answers_user_query,
            confidence=result.confidence,
            retry_recommendation=result.retry_recommendation,
            missing_requirements=result.missing_requirements,
            wrong_or_suspicious_parts=[*result.wrong_or_suspicious_parts, *result.readability_issues,
                                       *_publication_issues(result)],
            improvement_comments=result.improvement_comments,
            feedback_for_next_generation=result.feedback_for_next_generation,
        )

    @staticmethod
    def to_chart_analysis_record(
            *,
            query: str,
            result: VisualChartJudgeResult,
            used_fields: list[str],
    ) -> ChartGroundedAnalysisRecord:
        return ChartGroundedAnalysisRecord(
            user_query=query,
            used_fields=used_fields,
            chart_type=result.detected_chart_type,
            observed_facts=result.observed_facts,
            issues=[*result.missing_requirements, *result.wrong_or_suspicious_parts, *result.readability_issues,
                    *_publication_issues(result)],
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
            result: VisualChartJudgeResult,
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


def _clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _clean_list(values: list[str]) -> list[str]:
    return [str(item).strip() for item in values if str(item).strip()]


def _strip_large_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_large_data(item) for key, item in value.items() if key not in {"data", "datasets"}}
    if isinstance(value, list):
        return [_strip_large_data(item) for item in value]
    return value
