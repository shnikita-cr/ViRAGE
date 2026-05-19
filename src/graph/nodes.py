from __future__ import annotations

import json
from typing import Any

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.domain.models import AnalysisRubric, InsightsResult, PlotImageArtifact, SemanticFeedbackLoopSummary, StepLog, \
    VLMAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.observability import traceable
from src.services.chart_generator import ChartGeneratorService
from src.services.compact_data_profile import CompactDataProfileService
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
    SemanticChartJudgeAdapters,
    SemanticChartJudgeService,
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
        original = getattr(column, "original_name", None) or getattr(column, "name", "")
        safe = getattr(column, "safe_name", None) or original
        columns.append({
            "original_name": original,
            "safe_name": safe,
            "type": getattr(column, "dtype", "unknown"),
            "role": getattr(profile, "field_roles", {}).get(original,
                                                            getattr(profile, "field_roles", {}).get(safe, "unknown")),
            "missing_ratio": getattr(column, "missing_ratio", 0.0),
            "unique_count": getattr(column, "unique_count", 0),
            "min": getattr(column, "min_value", None),
            "max": getattr(column, "max_value", None),
            "sample_values": list(getattr(column, "sample_values", []) or []),
            "outlier_count": getattr(column, "outlier_count", 0),
            "outlier_ratio": getattr(column, "outlier_ratio", 0.0),
            "is_identifier": getattr(column, "is_identifier", False),
            "is_high_cardinality": getattr(column, "is_high_cardinality", False),
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
        "cleaning_hints": list(getattr(profile, "cleaning_hints", []) or []),
        "column_errors": list(getattr(profile, "column_errors", []) or []),
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


class PipelineNodes:
    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.query_request_analyzer = QueryRequestAnalyzerService()
        self.data_profiler = DataProfilerService()
        self.data_preparation = DataPreparationService()
        self.compact_data_profile = CompactDataProfileService()
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
        self.semantic_chart_judge = SemanticChartJudgeService()
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
        understanding = state.get("query_understanding")
        request = state.get("request_analysis")
        if understanding is not None:
            focus_areas.extend([understanding.task_type or "", understanding.analysis_goal or ""])
        if request is not None:
            focus_areas.extend(request.selected_fields)
        return AnalysisRubric(focus_areas=list(dict.fromkeys(item for item in focus_areas if item)))

    @traceable(name="virage.data_profiler")
    def data_profiler_node(self, state: PipelineState) -> dict:
        result = self.data_profiler.invoke(state["data_path"], runtime=self.runtime)
        compact_profile = None
        if bool(getattr(self.runtime.settings, "spec_generation_use_compact_profile", True)):
            compact_profile = self.compact_data_profile.invoke_from_profile(
                result,
                settings=self.runtime.settings,
            )
        artifact_paths = self._save(state, "data_profile", _data_profile_artifact_payload(result, self.runtime))
        return {
            "data_profile": result,
            "compact_data_profile": compact_profile,
            "stage": PipelineStage.DATA_PROFILING,
            "trace": self._trace(state, "data_profiler"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="data_profiler",
                title="Data profiling",
                summary=f"Rows={result.row_count}, Cols={result.col_count}",
                inputs=[state["data_path"]],
                outputs=[", ".join(result.likely_numeric_columns[:3]), ", ".join(result.likely_time_columns[:3])],
                details={
                    "artifact": artifact_paths["data_profile"],
                    "field_roles": result.field_roles,
                },
            ),
        }

    @traceable(name="virage.query_request_analysis")
    def query_request_analysis_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.query_request_analyzer.invoke(
            state["query"],
            state.get("user_context", {}),
            state["data_profile"],
            runtime=self.runtime,
            compact_data_profile=state.get("compact_data_profile"),
        )
        query_understanding = result.query_understanding
        request_analysis = result.request_analysis
        artifact_paths = self._save(state, "query_request_analysis", result.model_dump())
        return {
            "query_request_analysis": result.model_dump(),
            "query_understanding": query_understanding,
            "query_intent_bundle": query_understanding.to_intent_bundle(),
            "request_analysis": request_analysis,
            "chart_quality_requirements": result.chart_quality_requirements,
            "stage": PipelineStage.QUERY_REQUEST_ANALYSIS,
            "trace": self._trace(state, "query_request_analysis"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="query_request_analysis",
                title="Query and request analysis",
                summary=f"{query_understanding.intent}; fields={', '.join(request_analysis.selected_fields[:5])}",
                inputs=[state["query"], "data_profile"],
                outputs=[query_understanding.task_type or "unknown", *request_analysis.grounded_fields[:4]],
                details=self._stage_details(before) | {"artifact": artifact_paths["query_request_analysis"]},
            ),
        }

    @traceable(name="virage.data_preparation")
    def data_preparation_node(self, state: PipelineState) -> dict:
        result = self.data_preparation.invoke(
            state["data_path"],
            state["data_profile"],
            state["request_analysis"],
            state["run_id"],
            runtime=self.runtime,
            query_understanding=state.get("query_understanding"),
        )
        artifact_paths = self._save(state, "data_preparation", result.model_dump())
        compact_profile = None
        if bool(getattr(self.runtime.settings, "spec_generation_use_compact_profile", True)):
            compact_profile = self.compact_data_profile.invoke(
                state["data_profile"],
                result,
                state.get("request_analysis"),
                settings=self.runtime.settings,
            )
        return {
            "data_preparation": result,
            "compact_data_profile": compact_profile,
            "stage": PipelineStage.DATA_PREPARATION,
            "trace": self._trace(state, "data_preparation"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="data_preparation",
                title="Data preparation",
                summary=f"Prepared rows={result.row_count}",
                inputs=[state["data_path"]],
                outputs=result.operations[:5],
                details={
                    "artifact": artifact_paths["data_preparation"],
                    "prepared_path": result.output_path,
                },
            ),
        }

    @traceable(name="virage.visrag")
    def visrag_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.visrag.invoke(state["query_understanding"], state["request_analysis"], state["data_profile"],
                                    runtime=self.runtime)
        artifact_paths = self._save(state, "visrag", result.model_dump())
        candidate_set = result.candidate_spec_set
        selected = candidate_set.selected_candidate_spec if candidate_set else None
        return {
            "visrag": result,
            "candidate_spec_set": candidate_set,
            "analysis_rubric": self._analysis_rubric(state),
            "stage": PipelineStage.VISRAG,
            "trace": self._trace(state, "visrag"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="visrag",
                title="Spec retrieval",
                summary=selected.summary if selected else "no candidate",
                inputs=[state["query_understanding"].intent],
                outputs=[item.chart_family for item in (candidate_set.candidate_specs[:3] if candidate_set else [])],
                details=self._stage_details(before) | {"artifact": artifact_paths["visrag"],
                                                       "retrieval_query": result.retrieval_query},
            ),
        }

    @traceable(name="virage.chart_generator")
    def chart_generator_node(self, state: PipelineState) -> dict:
        technical_attempt = max(1, int(state.get("technical_attempt_number", 1) or 1))
        semantic_attempt = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        max_generation_attempts = max(1, int(self.runtime.settings.spec_generation_max_attempts))
        semantic_feedback_items = _merge_unique_texts(
            _manual_feedback_items(state.get("user_context", {})),
            list(state.get("semantic_feedback_items", [])),
        )
        semantic_chart_fact_history = list(state.get("semantic_chart_fact_history", []))
        technical_feedback = state.get("technical_retry_feedback") or {}

        result = self.chart_generator.invoke(
            state["data_preparation"],
            state["candidate_spec_set"],
            runtime=self.runtime,
            query=state["query"],
            data_profile=state.get("data_profile"),
            compact_data_profile=state.get("compact_data_profile"),
            request_analysis=state.get("request_analysis"),
            query_understanding=state.get("query_understanding"),
            visrag=state.get("visrag"),
            generation_attempt_number=technical_attempt,
            max_generation_attempts=max_generation_attempts,
            previous_validation_errors=list(technical_feedback.get("validation_errors") or []),
            previous_repair_hints=list(technical_feedback.get("repair_hints") or []),
            previous_invalid_spec=technical_feedback.get("invalid_spec"),
            previous_semantic_feedback=semantic_feedback_items,
            previous_chart_facts=semantic_chart_fact_history,
            chart_quality_requirements=list(state.get("chart_quality_requirements", [])),
        )
        artifact_paths = self._save(state, "vega_spec", _vega_spec_artifact_payload(result))
        selected = state["candidate_spec_set"].selected_candidate_spec if state.get("candidate_spec_set") else None
        return {
            "vega_spec": result,
            "technical_status": "generated",
            "semantic_feedback_items": semantic_feedback_items,
            "stage": PipelineStage.CHART_GENERATION,
            "trace": self._trace(state, "chart_generator"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="chart_generator",
                title="Chart generation",
                summary=f"Generated Vega-Lite specification; technical attempt {technical_attempt}, semantic attempt {semantic_attempt}",
                inputs=[selected.chart_family if selected else "", f"semantic_feedback={len(semantic_feedback_items)}"],
                outputs=[str((result.spec_without_runtime_data or result.spec_json).get("mark", ""))],
                details={"artifact": artifact_paths["vega_spec"], "spec": _vega_spec_artifact_payload(result)["spec"]},
            ),
        }

    @staticmethod
    def _validation_attempt_payload(attempt_number: int, vega_spec, validation_result) -> dict:
        return {
            "generation_attempt_number": attempt_number,
            "is_valid": validation_result.is_valid,
            "spec_json": vega_spec.spec_json,
            "validated_spec": validation_result.validated_spec,
            "validation_errors": validation_result.validation_errors,
            "repair_hints": validation_result.repair_hints,
        }

    @staticmethod
    def _validation_attempt_report(attempt_number: int, payload: dict) -> str:
        spec_json = json.dumps(payload.get("spec_json", {}), ensure_ascii=False, indent=2, default=str)
        validated_spec = json.dumps(payload.get("validated_spec", {}), ensure_ascii=False, indent=2, default=str)
        errors = payload.get("validation_errors") or []
        hints = payload.get("repair_hints") or []
        error_block = "\n".join(f"- {item}" for item in errors) or "- none"
        hint_block = "\n".join(f"- {item}" for item in hints) or "- none"
        return (
            f"# Spec validation attempt {attempt_number:03d}\n\n"
            f"## Status\n\nvalid = `{payload.get('is_valid')}`\n\n"
            f"## Vega-Lite spec sent to validator\n\n```json\n{spec_json}\n```\n\n"
            f"## Validator normalized spec\n\n```json\n{validated_spec}\n```\n\n"
            f"## Errors\n\n{error_block}\n\n"
            f"## Repair hints\n\n{hint_block}\n"
        )

    @traceable(name="virage.spec_validator")
    def spec_validator_node(self, state: PipelineState) -> dict:
        attempt_number = max(1, int(state.get("technical_attempt_number", 1) or 1))
        current_spec = state["vega_spec"]
        validation_result = self.spec_validator.invoke(current_spec)
        payload = self._validation_attempt_payload(attempt_number, current_spec, validation_result)
        artifact_paths = dict(state.get("artifact_paths", {}))
        artifact_paths = self._save_attempt_into(artifact_paths, state, "spec_validation", payload)

        return {
            "spec_validation": validation_result,
            "technical_status": "ok" if validation_result.is_valid else "validation_failed",
            "stage": PipelineStage.SPEC_VALIDATION,
            "trace": self._trace(state, "spec_validator"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="spec_validator",
                title="Specification validation",
                summary=f"valid={validation_result.is_valid}; technical attempt {attempt_number}",
                inputs=["vega_spec"],
                outputs=["validated_spec" if validation_result.is_valid else "validation_errors"],
                details={
                    "artifact": artifact_paths.get("spec_validation"),
                    "attempt_count": attempt_number,
                    "max_generation_attempts": int(self.runtime.settings.spec_generation_max_attempts),
                    "validated_spec": validation_result.validated_spec,
                    "validation_errors": validation_result.validation_errors,
                    "repair_hints": validation_result.repair_hints,
                },
            ),
        }

    @traceable(name="virage.technical_decision")
    def technical_decision_node(self, state: PipelineState) -> dict:
        validation = state["spec_validation"]
        attempt_number = max(1, int(state.get("technical_attempt_number", 1) or 1))
        max_attempts = max(1, int(self.runtime.settings.spec_generation_max_attempts))
        artifact_paths = dict(state.get("artifact_paths", {}))

        if validation.is_valid:
            summary = {
                "status": "ok",
                "succeeded": True,
                "completed_generation_attempts": attempt_number,
                "max_generation_attempts": max_attempts,
            }
            artifact_paths = self._save_attempt_into(artifact_paths, state, "technical_decision", summary)
            return {
                "technical_status": "ok",
                "stage": PipelineStage.SPEC_VALIDATION,
                "trace": self._trace(state, "technical_decision"),
                "artifact_paths": artifact_paths,
                "step_logs": self._append_log(
                    state,
                    stage="technical_decision",
                    title="Technical retry decision",
                    summary="technical validation accepted",
                    outputs=["ok"],
                    details=summary,
                ),
            }

        feedback = {
            "validation_errors": validation.validation_errors,
            "repair_hints": validation.repair_hints,
            "invalid_spec": state["vega_spec"].spec_json,
        }
        if attempt_number < max_attempts:
            summary = {
                "status": "retry",
                "succeeded": False,
                "completed_generation_attempts": attempt_number,
                "next_generation_attempt": attempt_number + 1,
                "max_generation_attempts": max_attempts,
                "validation_errors": validation.validation_errors,
                "repair_hints": validation.repair_hints,
            }
            artifact_paths = self._save_attempt_into(artifact_paths, state, "technical_decision", summary)
            return {
                "technical_status": "retry",
                "technical_attempt_number": attempt_number + 1,
                "technical_retry_feedback": feedback,
                "stage": PipelineStage.SPEC_VALIDATION,
                "trace": self._trace(state, "technical_decision_retry"),
                "artifact_paths": artifact_paths,
                "step_logs": self._append_log(
                    state,
                    stage="technical_decision",
                    title="Technical retry decision",
                    summary=f"retry spec generation {attempt_number + 1}/{max_attempts}",
                    outputs=["retry"],
                    details=summary,
                ),
            }

        summary = {
            "status": "failed",
            "succeeded": False,
            "completed_generation_attempts": attempt_number,
            "max_generation_attempts": max_attempts,
            "validation_errors": validation.validation_errors,
            "repair_hints": validation.repair_hints,
        }
        artifact_paths = self._save_attempt_into(artifact_paths, state, "technical_decision", summary)
        self.runtime.save_text_artifact(
            "errors/spec_validation_failed.txt",
            "\n".join(validation.validation_errors),
            run_id=state["run_id"],
            numbered=True,
        )
        raise RuntimeError(
            f"Specification validation failed after {max_attempts} graph-level generation attempt(s). "
            "See spec_validation_attempt_* artifacts for spec code, errors and repair hints."
        )

    @traceable(name="virage.vegalite_plot_drawing")
    def vegalite_plot_drawing_node(self, state: PipelineState) -> dict:
        result = self.vegalite_plot_drawing.invoke(state["spec_validation"], state["run_id"], runtime=self.runtime)
        artifact_paths = self._save(state, "plot_rendering", result.model_dump())
        return {
            "plot_rendering": result,
            "plot_image": result.plot_image.model_dump(),
            "stage": PipelineStage.PLOT_RENDERING,
            "trace": self._trace(state, "vegalite_plot_drawing"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="vegalite_plot_drawing",
                title="Plot rendering",
                summary=result.plot_image.image_path,
                inputs=["validated_spec"],
                outputs=result.render_notes[:5],
                details={"artifact": artifact_paths["plot_rendering"],
                         "rendered_scenegraph": result.rendered_scenegraph},
            ),
        }

    @traceable(name="virage.scenegraph_check")
    def scenegraph_check_node(self, state: PipelineState) -> dict:
        result = self.scenegraph_check.invoke(state["plot_rendering"])
        artifact_paths = self._save(state, "scenegraph_check", result.model_dump())
        return {
            "scenegraph_check": result,
            "stage": PipelineStage.SCENEGRAPH_CHECK,
            "trace": self._trace(state, "scenegraph_check"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="scenegraph_check",
                title="Scenegraph check",
                summary="Scenegraph inspected",
                inputs=["rendered_scenegraph"],
                outputs=result.notes[:5],
                details={"artifact": artifact_paths["scenegraph_check"], **result.model_dump()},
            ),
        }

    @traceable(name="virage.empty_chart_check")
    def empty_chart_check_node(self, state: PipelineState) -> dict:
        result = self.empty_chart_check.invoke(state["scenegraph_check"])
        artifact_paths = self._save(state, "empty_chart_check", result.model_dump())
        if result.empty_chart_signal:
            self.runtime.save_text_artifact("errors/empty_chart_detected.txt", result.empty_chart_status,
                                            run_id=state["run_id"], numbered=True)
            raise RuntimeError(
                "Rendered chart is empty or unusable. See run artifacts and errors directory for numbered details.")

        live_preview = _build_live_chart_preview_payload(state, result)

        return {
            "empty_chart_check": result,
            "stage": PipelineStage.EMPTY_CHART_CHECK,
            "trace": self._trace(state, "empty_chart_check"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="empty_chart_check",
                title="Empty chart check",
                summary=result.empty_chart_status,
                inputs=["scenegraph_status"],
                outputs=[str(result.empty_chart_signal), "live_preview_ready"],
                details={
                    "artifact": artifact_paths["empty_chart_check"],
                    "live_chart_preview": live_preview,
                    **result.model_dump(),
                },
            ),
        }

    @traceable(name="virage.spec_score")
    def spec_score_node(self, state: PipelineState) -> dict:
        if not self.runtime.settings.enable_spec_score:
            return {}
        ground_truth = state.get("user_context", {}).get("ground_truth_spec")
        if not isinstance(ground_truth, dict):
            return {}
        result = self.spec_score.invoke(
            state["spec_validation"],
            ground_truth,
            user_prompt=state.get("query"),
            empty_chart_check=state.get("empty_chart_check"),
        )
        artifact_paths = self._save(state, "structural_spec_metric", result.model_dump())
        return {
            "structural_spec_metric": result,
            "stage": PipelineStage.EVALUATION,
            "trace": self._trace(state, "spec_score"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="spec_score",
                title="Structural metric",
                summary=f"score={result.score:.3f}",
                inputs=["validated spec", "ground truth spec"],
                outputs=result.details[:3],
                details={"artifact": artifact_paths["structural_spec_metric"], **result.model_dump()},
            ),
        }

    @traceable(name="virage.semantic_loop_gate")
    def semantic_loop_gate_node(self, state: PipelineState) -> dict:
        if not self.runtime.settings.semantic_feedback_loop_enabled:
            summary = SemanticFeedbackLoopSummary(enabled=False, final_status="disabled")
            artifact_paths = self._save(state, "semantic_feedback_loop_summary", summary.model_dump())
            return {
                "semantic_status": "disabled",
                "semantic_feedback_loop_summary": summary,
                "stage": PipelineStage.EVALUATION,
                "trace": self._trace(state, "semantic_loop_gate_disabled"),
                "artifact_paths": artifact_paths,
                "step_logs": self._append_log(
                    state,
                    stage="semantic_loop_gate",
                    title="Semantic VLM loop",
                    summary="disabled",
                    outputs=["disabled"],
                    details={"artifact": artifact_paths["semantic_feedback_loop_summary"]},
                ),
            }
        mode = str(getattr(self.runtime.settings, "semantic_feedback_mode", "strict") or "strict")
        return {
            "semantic_status": "enabled",
            "semantic_feedback_mode": mode,
            "semantic_attempt_number": max(1, int(state.get("semantic_attempt_number", 1) or 1)),
            "stage": PipelineStage.EVALUATION,
            "trace": self._trace(state, "semantic_loop_gate_enabled"),
            "step_logs": self._append_log(
                state,
                stage="semantic_loop_gate",
                title="Semantic VLM loop",
                summary=f"enabled ({mode})",
                outputs=[mode],
            ),
        }

    @traceable(name="virage.semantic_chart_judge")
    def semantic_chart_judge_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        plot_image = PlotImageArtifact(**state["plot_image"])
        result = self.semantic_chart_judge.invoke(
            query=state["query"],
            plot_image=plot_image,
            vega_spec=state["vega_spec"],
            runtime=self.runtime,
            request_analysis=state.get("request_analysis"),
            data_profile=state.get("data_profile"),
            compact_data_profile=state.get("compact_data_profile"),
        )
        vlm_description = SemanticChartJudgeAdapters.to_vlm_description(result)
        chart_facts = SemanticChartJudgeAdapters.to_fact_summary(result)
        answer_judge = SemanticChartJudgeAdapters.to_answer_judge(result)
        retry_reasons = _semantic_retry_reasons(answer_judge)
        used_fields = []
        if state.get("request_analysis") is not None:
            used_fields = list(state["request_analysis"].selected_fields)
        chart_analysis = SemanticChartJudgeAdapters.to_chart_analysis_record(
            query=state["query"],
            result=result,
            used_fields=used_fields,
        )
        revision = SemanticChartJudgeAdapters.to_revision_record(
            attempt_number=attempt_number,
            query=state["query"],
            result=result,
            vega_spec=state["vega_spec"],
            rendered_png_path=state["plot_image"].get("image_path", ""),
            retry_reasons=retry_reasons,
        )
        semantic_payload = {
            "semantic_chart_judge": result.model_dump(),
            "chart_analysis": chart_analysis.model_dump(),
            "chart_feedback": answer_judge.model_dump(),
            "chart_revision_record": revision.model_dump(),
        }
        artifact_paths = self._save(state, "semantic_chart_judge", semantic_payload)
        history = [*state.get("semantic_chart_fact_history", []), chart_facts.model_dump()]
        return {
            "semantic_chart_judge": result,
            "vlm_chart_description": vlm_description,
            "chart_fact_summary": chart_facts,
            "semantic_chart_fact_history": history,
            "chart_answer_judge": answer_judge,
            "chart_analysis": chart_analysis,
            "chart_revision_record": revision,
            "semantic_retry_reasons": retry_reasons,
            "stage": PipelineStage.VERIFICATION,
            "trace": self._trace(state, "semantic_chart_judge"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="semantic_chart_judge",
                title="Strict semantic chart judge",
                summary=f"answers={result.answers_user_query}; recommendation={result.retry_recommendation}; confidence={result.confidence:.3f}",
                inputs=[state["plot_image"].get("image_path", ""), "user_query", "validated_spec"],
                outputs=[result.retry_recommendation, *retry_reasons[:2]],
                details=self._stage_details(before) | {
                    "artifact": artifact_paths["semantic_chart_judge"],
                    **result.model_dump(),
                },
            ),
        }

    @traceable(name="virage.vlm_chart_description")
    def vlm_chart_description_node(self, state: PipelineState) -> dict:
        plot_image = PlotImageArtifact(**state["plot_image"])
        result = self.vlm_chart_description.invoke(plot_image, runtime=self.runtime)
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        artifact_paths = self._save(state, f"semantic_attempt_{attempt_number:03d}_vlm_chart_description",
                                    result.model_dump())
        return {
            "vlm_chart_description": result,
            "stage": PipelineStage.VLM_ANALYSIS,
            "trace": self._trace(state, "vlm_chart_description"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="vlm_chart_description",
                title="PNG-only chart description",
                summary="described rendered PNG only",
                inputs=[state["plot_image"]["image_path"]],
                outputs=[result.detected_chart_type or "unknown"],
                details={"artifact": artifact_paths[f"semantic_attempt_{attempt_number:03d}_vlm_chart_description"],
                         **result.model_dump()},
            ),
        }

    @traceable(name="virage.chart_fact_summary")
    def chart_fact_summary_node(self, state: PipelineState) -> dict:
        result = self.chart_fact_summary.invoke(state["vlm_chart_description"], runtime=self.runtime)
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        artifact_paths = self._save(state, f"semantic_attempt_{attempt_number:03d}_chart_fact_summary",
                                    result.model_dump())
        history = [*state.get("semantic_chart_fact_history", []), result.model_dump()]
        return {
            "chart_fact_summary": result,
            "semantic_chart_fact_history": history,
            "stage": PipelineStage.VLM_ANALYSIS,
            "trace": self._trace(state, "chart_fact_summary"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="chart_fact_summary",
                title="Chart fact summary",
                summary=f"{len(result.facts)} facts",
                inputs=["png-only VLM description"],
                outputs=result.facts[:3],
                details={"artifact": artifact_paths[f"semantic_attempt_{attempt_number:03d}_chart_fact_summary"],
                         **result.model_dump()},
            ),
        }

    @traceable(name="virage.chart_answer_judge")
    def chart_answer_judge_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.chart_answer_judge.invoke(
            query=state["query"],
            chart_facts=state["chart_fact_summary"],
            runtime=self.runtime,
            request_analysis=state.get("request_analysis"),
        )
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        artifact_paths = self._save(state, f"semantic_attempt_{attempt_number:03d}_answer_judge", result.model_dump())
        judge_model_calls = [item.model_dump() for item in self.runtime.model_call_logs[before:]]
        if judge_model_calls:
            artifact_paths = self._save_into(
                artifact_paths,
                state["run_id"],
                f"semantic_attempt_{attempt_number:03d}_answer_judge_model_calls",
                judge_model_calls,
            )
        return {
            "chart_answer_judge": result,
            "stage": PipelineStage.VERIFICATION,
            "trace": self._trace(state, "chart_answer_judge"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="chart_answer_judge",
                title="Semantic answer judge",
                summary=f"answers={result.answers_user_query}; confidence={result.confidence:.3f}",
                inputs=["user_query", "chart_facts"],
                outputs=[result.retry_recommendation],
                details={"artifact": artifact_paths[f"semantic_attempt_{attempt_number:03d}_answer_judge"],
                         **result.model_dump()},
            ),
        }

    @traceable(name="virage.semantic_decision")
    def semantic_decision_node(self, state: PipelineState) -> dict:
        judge = state["chart_answer_judge"]
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        max_attempts = max(1, int(self.runtime.settings.semantic_feedback_max_attempts))
        min_confidence = float(self.runtime.settings.semantic_feedback_min_accept_confidence)
        actionable_feedback = _actionable_semantic_feedback(judge)
        retry_reasons = _semantic_retry_reasons(judge)
        accepted = bool(
            judge.answers_user_query
            and judge.confidence >= min_confidence
            and (judge.retry_recommendation == "accept" or not actionable_feedback)
        )
        artifact_paths = dict(state.get("artifact_paths", {}))

        if accepted:
            summary = SemanticFeedbackLoopSummary(
                enabled=True,
                max_attempts=max_attempts,
                attempt_count=attempt_number,
                retry_count=max(0, attempt_number - 1),
                accepted=True,
                accepted_attempt=attempt_number,
                final_status="accepted",
                final_confidence=judge.confidence,
                saved_feedback_count=len(state.get("visual_feedback_examples", [])),
                missing_requirements=judge.missing_requirements,
                improvement_comments=judge.improvement_comments,
                feedback_corpus_path=str(self.runtime.settings.semantic_feedback_corpus_path),
            )
            artifact_paths = self._save_into(artifact_paths, state["run_id"], "semantic_feedback_loop_summary",
                                             summary.model_dump())
            summary.summary_artifact_path = artifact_paths["semantic_feedback_loop_summary"]
            return {
                "semantic_status": "accepted",
                "semantic_feedback_loop_summary": summary,
                "stage": PipelineStage.VERIFICATION,
                "trace": self._trace(state, "semantic_decision_accept"),
                "artifact_paths": artifact_paths,
                "step_logs": self._append_log(
                    state,
                    stage="semantic_decision",
                    title="Semantic retry decision",
                    summary="accepted",
                    outputs=["accept"],
                    details=summary.model_dump(),
                ),
            }

        feedback_text = _semantic_feedback_text(judge)
        if not feedback_text and retry_reasons:
            feedback_text = "\n".join(retry_reasons)
        feedback_items = [*state.get("semantic_feedback_items", [])]
        if feedback_text:
            feedback_items.append(feedback_text)

        if retry_reasons and attempt_number < max_attempts:
            summary = {
                "status": "retry",
                "attempt_number": attempt_number,
                "next_semantic_attempt": attempt_number + 1,
                "max_attempts": max_attempts,
                "confidence": judge.confidence,
                "answers_user_query": judge.answers_user_query,
                "retry_recommendation": judge.retry_recommendation,
                "actionable_feedback_exists": bool(actionable_feedback),
                "retry_reasons": retry_reasons,
                "decision_reason": "retry_with_concrete_reasons",
                "missing_requirements": judge.missing_requirements,
                "wrong_or_suspicious_parts": judge.wrong_or_suspicious_parts,
                "improvement_comments": judge.improvement_comments,
                "feedback_for_next_generation": feedback_text,
            }
            artifact_paths = self._save_into(artifact_paths, state["run_id"],
                                             "semantic_decision", summary)
            return {
                "semantic_status": "retry",
                "semantic_attempt_number": attempt_number + 1,
                "technical_attempt_number": 1,
                "technical_retry_feedback": {},
                "semantic_feedback_items": feedback_items,
                "semantic_retry_feedback": feedback_text,
                "semantic_retry_reasons": retry_reasons,
                "stage": PipelineStage.VERIFICATION,
                "trace": self._trace(state, "semantic_decision_retry"),
                "artifact_paths": artifact_paths,
                "step_logs": self._append_log(
                    state,
                    stage="semantic_decision",
                    title="Semantic retry decision",
                    summary=f"retry spec generation for semantic attempt {attempt_number + 1}/{max_attempts}",
                    outputs=["retry"],
                    details=summary,
                ),
            }

        final_examples = list(state.get("visual_feedback_examples", []))
        final_feedback_example_path = ""
        final_corpus_path = ""
        try:
            final_example = self.feedback_corpus_writer.build_example(
                run_id=state["run_id"],
                attempt_number=attempt_number,
                query=state["query"],
                vega_spec=state["vega_spec"],
                rendered_png_path=state["plot_image"]["image_path"],
                vlm_description=state["vlm_chart_description"],
                chart_facts=state["chart_fact_summary"],
                judge_result=judge,
                request_analysis=state.get("request_analysis"),
            )
            artifact_paths = self._save_into(
                artifact_paths,
                state["run_id"],
                f"semantic_attempt_{attempt_number:03d}_feedback_example",
                final_example.model_dump(),
            )
            final_feedback_example_path = artifact_paths.get(f"semantic_attempt_{attempt_number:03d}_feedback_example",
                                                             "")
            final_examples.append(final_example)
            if self.runtime.settings.semantic_feedback_save_rejected_specs:
                final_corpus_path = self.feedback_corpus_writer.append_to_corpus(final_example, self.runtime)
        except Exception as exc:
            artifact_paths = self._save_into(
                artifact_paths,
                state["run_id"],
                f"semantic_attempt_{attempt_number:03d}_feedback_write_error",
                {"error": f"{type(exc).__name__}: {exc}"},
            )

        summary = SemanticFeedbackLoopSummary(
            enabled=True,
            max_attempts=max_attempts,
            attempt_count=attempt_number,
            retry_count=max(0, attempt_number - 1),
            accepted=False,
            final_status="failed",
            final_confidence=judge.confidence,
            saved_feedback_count=len(final_examples),
            missing_requirements=judge.missing_requirements,
            improvement_comments=judge.improvement_comments,
            feedback_corpus_path=final_corpus_path or str(self.runtime.settings.semantic_feedback_corpus_path),
        )
        artifact_paths = self._save_into(artifact_paths, state["run_id"], "semantic_feedback_loop_summary",
                                         summary.model_dump())
        summary.summary_artifact_path = artifact_paths["semantic_feedback_loop_summary"]
        return {
            "semantic_status": "failed",
            "visual_feedback_examples": final_examples,
            "semantic_feedback_loop_summary": summary,
            "stage": PipelineStage.VERIFICATION,
            "trace": self._trace(state, "semantic_decision_failed"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="semantic_decision",
                title="Semantic retry decision",
                summary="max semantic attempts reached; continuing with warnings",
                outputs=["failed"],
                details=summary.model_dump(),
            ),
        }

    @traceable(name="virage.feedback_corpus_writer")
    def feedback_corpus_writer_node(self, state: PipelineState) -> dict:
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        example = self.feedback_corpus_writer.build_example(
            run_id=state["run_id"],
            attempt_number=attempt_number,
            query=state["query"],
            vega_spec=state["vega_spec"],
            rendered_png_path=state["plot_image"]["image_path"],
            vlm_description=state["vlm_chart_description"],
            chart_facts=state["chart_fact_summary"],
            judge_result=state["chart_answer_judge"],
            request_analysis=state.get("request_analysis"),
        )
        artifact_paths = self._save(state, f"semantic_attempt_{attempt_number:03d}_feedback_example",
                                    example.model_dump())
        corpus_path = ""
        if self.runtime.settings.semantic_feedback_save_rejected_specs:
            corpus_path = self.feedback_corpus_writer.append_to_corpus(example, self.runtime)
        feedback_block = example.feedback_for_next_generation
        examples = [*state.get("visual_feedback_examples", []), example]
        feedback_items = list(state.get("semantic_feedback_items", []))
        if feedback_block and (not feedback_items or feedback_items[-1] != feedback_block):
            feedback_items.append(feedback_block)
        return {
            "visual_feedback_examples": examples,
            "semantic_feedback_items": feedback_items,
            "stage": PipelineStage.VERIFICATION,
            "trace": self._trace(state, "feedback_corpus_writer"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="feedback_corpus_writer",
                title="Feedback corpus writer",
                summary=f"saved feedback example; corpus={bool(corpus_path)}",
                inputs=["spec", "comments"],
                outputs=[corpus_path or "artifact-only"],
                details={"artifact": artifact_paths[f"semantic_attempt_{attempt_number:03d}_feedback_example"],
                         "corpus_path": corpus_path},
            ),
        }

    @traceable(name="virage.vlm_chart_analysis")
    def vlm_analysis_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        if state.get("semantic_status") == "failed":
            result = VLMAnalysisResult(
                summary="Chart-grounded analysis was skipped because semantic chart validation failed.",
                key_findings=[],
                caveats=["The chart was not accepted by the semantic judge."],
                suggested_followup_questions=[],
                visual_observations=[],
                extracted_visual_facts=[],
                confidence=0.0,
            )
            insights = InsightsResult(final_insights=[])
            artifact_paths = self._save(state, "vlm_analysis", result.model_dump())
            return {
                "vlm_analysis": result,
                "insights": insights,
                "stage": PipelineStage.VLM_ANALYSIS,
                "trace": self._trace(state, "vlm_chart_analysis_skipped_semantic_failed"),
                "artifact_paths": artifact_paths,
                "step_logs": self._append_log(
                    state,
                    stage="vlm_chart_analysis",
                    title="Chart-grounded VLM analysis",
                    summary="skipped because semantic judge did not accept the chart",
                    outputs=["skipped"],
                    details={"artifact": artifact_paths["vlm_analysis"], **result.model_dump()},
                ),
            }

        plot_image = PlotImageArtifact(**state["plot_image"])
        try:
            result = self.vlm_analysis.invoke(plot_image, state["analysis_rubric"], runtime=self.runtime)
        except Exception as exc:
            if not bool(getattr(self.runtime.settings, "vlm_fail_soft", True)):
                raise
            result = VLMAnalysisResult(
                summary=f"Chart-grounded analysis skipped after {type(exc).__name__}: {exc}",
                key_findings=[],
                caveats=["The VLM analysis model was unavailable."],
                suggested_followup_questions=[],
                visual_observations=[f"VLM analysis skipped after {type(exc).__name__}: {exc}"],
                extracted_visual_facts=[],
                confidence=0.0,
            )
            insights = InsightsResult(final_insights=[])
            artifact_paths = self._save(state, "vlm_analysis", result.model_dump())
            return {
                "vlm_analysis": result,
                "insights": insights,
                "stage": PipelineStage.VLM_ANALYSIS,
                "trace": self._trace(state, "vlm_chart_analysis_skipped"),
                "artifact_paths": artifact_paths,
                "step_logs": self._append_log(
                    state,
                    stage="vlm_chart_analysis",
                    title="Chart-grounded VLM analysis",
                    summary="VLM unavailable; continued without user insights",
                    inputs=[state["plot_image"]["image_path"]],
                    outputs=result.caveats[:1],
                    details=self._stage_details(before) | {
                        "artifact": artifact_paths["vlm_analysis"],
                        "error_type": "vlm_unavailable",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                ),
            }
        insights = InsightsResult(final_insights=[*result.key_findings] or ([result.summary] if result.summary else []))
        artifact_paths = self._save(state, "vlm_analysis", result.model_dump())
        return {
            "vlm_analysis": result,
            "insights": insights,
            "stage": PipelineStage.VLM_ANALYSIS,
            "trace": self._trace(state, "vlm_chart_analysis"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="vlm_chart_analysis",
                title="Chart-grounded VLM analysis",
                summary=result.summary or f"{len(result.key_findings)} findings",
                inputs=[state["plot_image"]["image_path"]],
                outputs=result.key_findings[:3] or result.visual_observations[:3],
                details=self._stage_details(before) | {"artifact": artifact_paths["vlm_analysis"],
                                                       **result.model_dump()},
            ),
        }

    @traceable(name="virage.evaluation_summary")
    def evaluation_summary_node(self, state: PipelineState) -> dict:
        if not self.runtime.settings.enable_evaluation_summary:
            return {}
        technical_attempt = max(1, int(state.get("technical_attempt_number", 1) or 1))
        result = self.evaluation_summary.invoke(
            state.get("structural_spec_metric"),
            state["empty_chart_check"],
            state.get("insights"),
            technical_status=str(state.get("technical_status") or "unknown"),
            semantic_status=str(state.get("semantic_status") or "unknown"),
            semantic_summary=state.get("semantic_feedback_loop_summary"),
            technical_retry_count=max(0, technical_attempt - 1),
            benchmark_scores={
                "spec_score": getattr(state.get("structural_spec_metric"), "score", None),
                "vision_score": None,
            },
        )
        artifact_paths = self._save(state, "evaluation_summary", result.model_dump())
        return {
            "evaluation_summary": result,
            "stage": PipelineStage.EVALUATION,
            "trace": self._trace(state, "evaluation_summary"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="evaluation_summary",
                title="Evaluation summary",
                summary="Aggregated metrics and verification results",
                inputs=["metrics", "verification"],
                outputs=[str(result.benchmark_report)],
                details={"artifact": artifact_paths["evaluation_summary"], **result.model_dump()},
            ),
        }
