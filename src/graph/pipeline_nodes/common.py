from __future__ import annotations

import json
from typing import Any

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.domain.models import AnalysisRubric, PlotImageArtifact, SemanticFeedbackLoopSummary, StepLog, \
    VLMAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.observability import traceable
from src.services.chart_generator import ChartGeneratorService
from src.services.data_preparation import DataPreparationService
from src.services.data_profiler import DataProfilerService
from src.services.empty_chart_check import EmptyChartCheckService
from src.services.evaluation_summary import EvaluationSummaryService
from src.services.query_request_analyzer import QueryRequestAnalyzerService
from src.services.scenegraph_check import ScenegraphCheckService
from src.services.spec_score import SpecScoreService
from src.services.spec_validator import SpecValidatorService
from src.services.vegalite_plot_drawing import VegaLitePlotDrawingService
from src.services.visrag import VisRAGService
from src.services.visual_feedback import (
    ChartAnswerJudgeService,
    ChartFactSummaryService,
    FeedbackCorpusWriterService,
    VisualChartJudgeAdapters,
    VisualChartJudgeService,
    VLMChartDescriptionService,
)
from src.services.vlm_analysis import VLMAnalysisService


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _merge_unique_texts(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for item in group:
            text = _clean_text(item)
            if text and text not in seen:
                seen.add(text)
                result.append(text)
    return result


def _manual_feedback_items(user_context: Any) -> list[str]:
    if not isinstance(user_context, dict):
        return []
    values: list[str] = []
    for key in (
            "manual_feedback", "manual_semantic_feedback", "user_chart_feedback", "user_feedback_for_next_generation"):
        value = user_context.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if str(item).strip())
    return _merge_unique_texts(values)


def _actionable_semantic_feedback(judge: Any) -> list[str]:
    return _merge_unique_texts(
        [getattr(judge, "feedback_for_next_generation", "")],
        list(getattr(judge, "missing_requirements", []) or []),
        list(getattr(judge, "wrong_or_suspicious_parts", []) or []),
        list(getattr(judge, "improvement_comments", []) or []),
    )


def _semantic_retry_reasons(judge: Any) -> list[str]:
    reasons = _actionable_semantic_feedback(judge)
    if bool(getattr(judge, "answers_user_query", False)) is False:
        reasons = _merge_unique_texts(["The chart does not fully answer the user query."], reasons)
    if str(getattr(judge, "retry_recommendation", "")).strip().lower() == "reject":
        reasons = _merge_unique_texts(["The semantic judge rejected the rendered chart."], reasons)
    return reasons


def _semantic_feedback_text(judge: Any) -> str:
    direct = _clean_text(getattr(judge, "feedback_for_next_generation", ""))
    if direct:
        return direct
    return "\n".join(_actionable_semantic_feedback(judge))


def _merge_generation_artifacts(
        artifact_paths: dict[str, str],
        generation_artifacts: dict[str, str],
        *,
        semantic_attempt: int,
        technical_attempt: int,
) -> dict[str, str]:
    if not generation_artifacts:
        return artifact_paths
    merged = dict(artifact_paths)
    for key, value in generation_artifacts.items():
        stable_key = f"semantic_{semantic_attempt:03d}_technical_{technical_attempt:03d}_{key}"
        merged[stable_key] = value
    return merged


def _model_dump_or_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(value, dict):
        return value
    return {"value": value}


def _build_live_chart_preview_payload(state: PipelineState, empty_chart_check: Any) -> dict[str, Any]:
    plot_image = state.get("plot_image")
    if hasattr(plot_image, "model_dump"):
        plot_image_payload = plot_image.model_dump()
    elif isinstance(plot_image, dict):
        plot_image_payload = dict(plot_image)
    else:
        plot_image_payload = {}

    spec_validation = state.get("spec_validation")
    data_preparation = state.get("data_preparation")
    scenegraph_check = state.get("scenegraph_check")

    return {
        "status": "ready_after_technical_validation",
        "image_path": str(plot_image_payload.get("image_path") or ""),
        "image": plot_image_payload,
        "validated_spec": _model_dump_or_dict(spec_validation).get("validated_spec", {}),
        "data_path": state.get("data_path", ""),
        "prepared_data_path": _model_dump_or_dict(data_preparation).get("output_path", ""),
        "semantic_attempt_number": int(state.get("semantic_attempt_number") or 1),
        "technical_attempt_number": int(state.get("technical_attempt_number") or 1),
        "scenegraph_check": _model_dump_or_dict(scenegraph_check),
        "empty_chart_check": _model_dump_or_dict(empty_chart_check),
    }


def _data_profile_artifact_payload(profile: Any, runtime: RuntimeContext) -> dict[str, Any]:
    columns = []
    for column in getattr(profile, "columns", []) or []:
        original = getattr(column, "name", "")
        safe = getattr(column, "safe_name", None) or original
        columns.append({
            "original_name": original,
            "safe_name": safe,
            "type": getattr(column, "dtype", "unknown"),
            "role": getattr(column, "role", "unknown"),
            "missing_ratio": getattr(column, "missing_ratio", 0.0),
            "unique_count": getattr(column, "unique_count", 0),
            "min": getattr(column, "min_value", None),
            "max": getattr(column, "max_value", None),
            "sample_values": list(getattr(column, "sample_values", []) or []),
            "outlier_count": getattr(column, "outlier_count", 0),
            "outlier_ratio": getattr(column, "outlier_ratio", 0.0),
            "is_identifier": getattr(column, "is_identifier", False),
            "is_high_cardinality": getattr(column, "is_high_cardinality", False),
            "raw_dtype": getattr(column, "raw_dtype", None),
            "missing_like_ratio": getattr(column, "missing_like_ratio", 0.0),
            "quality_flags": list(getattr(column, "quality_flags", []) or []),
            "preparation_hints": list(getattr(column, "preparation_hints", []) or []),
        })
    return {
        "row_count": getattr(profile, "row_count", 0),
        "column_count": getattr(profile, "col_count", 0),
        "profile_status": getattr(profile, "profile_status", "ok"),
        "data_complexity": getattr(profile, "data_complexity", None),
        "sample_strategy": getattr(runtime.settings, "data_profile_sample_strategy", "random"),
        "sample_seed": int(getattr(runtime.settings, "data_profile_sample_seed", 42)),
        "sample_size": int(getattr(runtime.settings, "data_profile_sample_size", 10)),
        "columns": columns,
        "quality_notes": list(getattr(profile, "quality_notes", []) or []),
        "complexity_hints": list(getattr(profile, "complexity_hints", []) or []),
        "errors": list(getattr(profile, "errors", []) or []),
    }


def _vega_spec_artifact_payload(spec_artifact: Any) -> dict[str, Any]:
    spec = getattr(spec_artifact, "spec_without_runtime_data", None) or getattr(spec_artifact, "spec_json", {}) or {}
    if isinstance(spec, dict):
        spec = {key: value for key, value in spec.items() if key not in {"data", "datasets"}}
    return {
        "spec": spec,
        "version": getattr(spec_artifact, "version", None),
        "generation_backend": getattr(spec_artifact, "generation_backend", None),
        "generation_explanation": getattr(spec_artifact, "generation_explanation", None),
        "generation_warnings": list(getattr(spec_artifact, "generation_warnings", []) or []),
    }




class BasePipelineNodes:
    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.query_request_analyzer = QueryRequestAnalyzerService()
        self.data_profiler = DataProfilerService()
        self.data_preparation = DataPreparationService()
        self.visrag = VisRAGService()
        self.chart_generator = ChartGeneratorService()
        self.spec_validator = SpecValidatorService()
        self.vegalite_plot_drawing = VegaLitePlotDrawingService()
        self.scenegraph_check = ScenegraphCheckService()
        self.empty_chart_check = EmptyChartCheckService()
        self.vlm_analysis = VLMAnalysisService()
        self.spec_score = SpecScoreService()
        self.evaluation_summary = EvaluationSummaryService()
        self.vlm_chart_description = VLMChartDescriptionService()
        self.chart_fact_summary = ChartFactSummaryService()
        self.chart_answer_judge = ChartAnswerJudgeService()
        self.visual_chart_judge = VisualChartJudgeService()
        self.feedback_corpus_writer = FeedbackCorpusWriterService()

    @staticmethod
    def _trace(state: PipelineState, label: str) -> list[str]:
        return [*state.get("trace", []), label]

    def _stage_details(self, before: int) -> dict:
        logs = self.runtime.model_call_logs[before:]
        return {
            "model_calls": [item.model_dump() for item in logs],
            "stage_token_usage": {
                "prompt_tokens": sum(item.token_usage.prompt_tokens for item in logs),
                "completion_tokens": sum(item.token_usage.completion_tokens for item in logs),
                "total_tokens": sum(item.token_usage.total_tokens for item in logs),
            },
        }

    def _append_log(
            self,
            state: PipelineState,
            *,
            stage: str,
            title: str,
            summary: str,
            inputs: list[str] | None = None,
            outputs: list[str] | None = None,
            details: dict | None = None,
    ) -> list[StepLog]:
        log = StepLog(stage=stage, title=title, summary=summary, inputs=inputs or [], outputs=outputs or [],
                      details=details or {})
        self.runtime.emit_step(log)
        return [*state.get("step_logs", []), log]

    _ATTEMPT_AWARE_NODE_ARTIFACTS = {
        "vega_spec",
        "spec_validation",
        "technical_decision",
        "plot_rendering",
        "scenegraph_check",
        "empty_chart_check",
        "structural_spec_metric",
        "visual_chart_judge",
        "semantic_chart_judge",
        "semantic_decision",
        "feedback_corpus_writer",
        "vlm_analysis",
        "evaluation_summary",
    }

    @classmethod
    def _node_artifact_name(cls, state: PipelineState, name: str) -> str:
        if name.startswith("semantic_attempt_") or name.endswith("_attempt"):
            return name
        if name not in cls._ATTEMPT_AWARE_NODE_ARTIFACTS:
            return name
        semantic_attempt = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        technical_attempt = max(1, int(state.get("technical_attempt_number", 1) or 1))
        return f"{name}_semantic_{semantic_attempt:03d}_technical_{technical_attempt:03d}"

    def _save(self, state: PipelineState, name: str, payload: object) -> dict[str, str]:
        artifact_name = self._node_artifact_name(state, name)
        path = self.runtime.save_json_artifact(f"nodes/{artifact_name}.json", payload, run_id=state["run_id"],
                                               numbered=True)
        return {**state.get("artifact_paths", {}), name: path, artifact_name: path}

    def _save_into(self, artifact_paths: dict[str, str], run_id: str, name: str, payload: object) -> dict[str, str]:
        path = self.runtime.save_json_artifact(f"nodes/{name}.json", payload, run_id=run_id, numbered=True)
        return {**artifact_paths, name: path}

    def _save_attempt_into(self, artifact_paths: dict[str, str], state: PipelineState, name: str, payload: object) -> \
            dict[str, str]:
        artifact_name = self._node_artifact_name(state, name)
        path = self.runtime.save_json_artifact(f"nodes/{artifact_name}.json", payload, run_id=state["run_id"],
                                               numbered=True)
        return {**artifact_paths, name: path, artifact_name: path}

    def _save_text_into(self, artifact_paths: dict[str, str], run_id: str, name: str, text: str) -> dict[str, str]:
        path = self.runtime.save_text_artifact(f"nodes/{name}.md", text, run_id=run_id, numbered=True)
        return {**artifact_paths, name: path}

    @staticmethod
    def _analysis_rubric(state: PipelineState) -> AnalysisRubric:
        focus_areas: list[str] = []
        analysis = state.get("query_request_analysis")
        if analysis is not None:
            focus_areas.extend([analysis.analysis_task or "", analysis.normalized_query or ""])
            focus_areas.extend(list(analysis.selected_fields or []))
        return AnalysisRubric(focus_areas=list(dict.fromkeys(item for item in focus_areas if item)))

__all__ = [
    'BasePipelineNodes',
    '_clean_text',
    '_merge_unique_texts',
    '_manual_feedback_items',
    '_actionable_semantic_feedback',
    '_semantic_retry_reasons',
    '_semantic_feedback_text',
    '_merge_generation_artifacts',
    '_model_dump_or_dict',
    '_build_live_chart_preview_payload',
    '_data_profile_artifact_payload',
    '_vega_spec_artifact_payload',
    'Any',
    'PipelineState',
    'PipelineStage',
    'AnalysisRubric',
    'PlotImageArtifact',
    'SemanticFeedbackLoopSummary',
    'StepLog',
    'VLMAnalysisResult',
    'traceable',
]
