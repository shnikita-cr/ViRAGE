from __future__ import annotations

import json

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.domain.models import AnalysisRubric, PlotImageArtifact, SemanticFeedbackLoopSummary, StepLog
from src.infrastructure.runtime import RuntimeContext
from src.observability import traceable
from src.services.chart_generator import ChartGeneratorService
from src.services.data_preparation import DataPreparationService
from src.services.data_profiler import DataProfilerService
from src.services.empty_chart_check import EmptyChartCheckService
from src.services.evaluation_summary import EvaluationSummaryService
from src.services.fact_extractor import FactExtractorService
from src.services.insights import InsightsService
from src.services.query_understanding import QueryUnderstandingService
from src.services.reasoner import ReasonerService
from src.services.request_analyzer import RequestAnalyzerService
from src.services.scenegraph_check import ScenegraphCheckService
from src.services.spec_score import SpecScoreService
from src.services.spec_validator import SpecValidatorService
from src.services.vegalite_plot_drawing import VegaLitePlotDrawingService
from src.services.vision_score import VisionScoreService
from src.services.visrag import VisRAGService
from src.services.vlm_analysis import VLMAnalysisService
from src.services.visual_feedback import (
    ChartAnswerJudgeService,
    ChartFactSummaryService,
    FeedbackCorpusWriterService,
    VLMChartDescriptionService,
)


class PipelineNodes:
    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.query_understanding = QueryUnderstandingService()
        self.data_profiler = DataProfilerService()
        self.request_analyzer = RequestAnalyzerService()
        self.data_preparation = DataPreparationService()
        self.visrag = VisRAGService()
        self.chart_generator = ChartGeneratorService()
        self.spec_validator = SpecValidatorService()
        self.vegalite_plot_drawing = VegaLitePlotDrawingService()
        self.scenegraph_check = ScenegraphCheckService()
        self.empty_chart_check = EmptyChartCheckService()
        self.vlm_analysis = VLMAnalysisService()
        self.fact_extractor = FactExtractorService()
        self.reasoner = ReasonerService()
        self.insights = InsightsService()
        self.spec_score = SpecScoreService()
        self.vision_score = VisionScoreService()
        self.evaluation_summary = EvaluationSummaryService()
        self.vlm_chart_description = VLMChartDescriptionService()
        self.chart_fact_summary = ChartFactSummaryService()
        self.chart_answer_judge = ChartAnswerJudgeService()
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

    def _save(self, state: PipelineState, name: str, payload: object) -> dict[str, str]:
        path = self.runtime.save_json_artifact(f"artifacts/{name}.json", payload, run_id=state["run_id"], numbered=True)
        return {**state.get("artifact_paths", {}), name: path}

    def _save_into(self, artifact_paths: dict[str, str], run_id: str, name: str, payload: object) -> dict[str, str]:
        path = self.runtime.save_json_artifact(f"artifacts/{name}.json", payload, run_id=run_id, numbered=True)
        return {**artifact_paths, name: path}

    def _save_text_into(self, artifact_paths: dict[str, str], run_id: str, name: str, text: str) -> dict[str, str]:
        path = self.runtime.save_text_artifact(f"artifacts/{name}.md", text, run_id=run_id, numbered=True)
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

    @traceable(name="virage.query_understanding")
    def query_understanding_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.query_understanding.invoke(
            state["query"],
            state.get("user_context", {}),
            runtime=self.runtime,
            data_profile=state.get("data_profile"),
        )
        artifact_paths = self._save(state, "query_understanding", result.model_dump())
        return {
            "query_understanding": result,
            "query_intent_bundle": result.to_intent_bundle(),
            "stage": PipelineStage.QUERY_UNDERSTANDING,
            "trace": self._trace(state, "query_understanding"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="query_understanding",
                title="Query understanding",
                summary=result.intent,
                inputs=[state["query"], "data_profile" if state.get("data_profile") else "no_data_profile"],
                outputs=[result.task_type or "unknown", result.analysis_goal or ""],
                details=self._stage_details(before) | {"artifact": artifact_paths["query_understanding"]},
            ),
        }

    @traceable(name="virage.data_profiler")
    def data_profiler_node(self, state: PipelineState) -> dict:
        result = self.data_profiler.invoke(state["data_path"], runtime=self.runtime)
        artifact_paths = self._save(state, "data_profile", result.model_dump())
        return {
            "data_profile": result,
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
                details={"artifact": artifact_paths["data_profile"], "field_roles": result.field_roles},
            ),
        }

    @traceable(name="virage.request_analyzer")
    def request_analyzer_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.request_analyzer.invoke(state["query"], state["query_understanding"], state["data_profile"],
                                              runtime=self.runtime)
        artifact_paths = self._save(state, "request_analysis", result.model_dump())
        return {
            "request_analysis": result,
            "stage": PipelineStage.REQUEST_ANALYSIS,
            "trace": self._trace(state, "request_analyzer"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="request_analyzer",
                title="Request analysis",
                summary=f"Selected fields: {', '.join(result.selected_fields)}",
                inputs=[state["query"]],
                outputs=result.grounded_fields[:5],
                details=self._stage_details(before) | {"artifact": artifact_paths["request_analysis"]},
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
        return {
            "data_preparation": result,
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
                details={"artifact": artifact_paths["data_preparation"], "prepared_path": result.output_path},
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
        semantic_feedback_items = list(state.get("semantic_feedback_items", []))
        semantic_chart_fact_history = list(state.get("semantic_chart_fact_history", []))
        technical_feedback = state.get("technical_retry_feedback") or {}

        result = self.chart_generator.invoke(
            state["data_preparation"],
            state["candidate_spec_set"],
            runtime=self.runtime,
            query=state["query"],
            data_profile=state.get("data_profile"),
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
        )
        artifact_paths = self._save(state, f"vega_spec_technical_{technical_attempt:03d}_semantic_{semantic_attempt:03d}", result.model_dump())
        selected = state["candidate_spec_set"].selected_candidate_spec if state.get("candidate_spec_set") else None
        return {
            "vega_spec": result,
            "technical_status": "generated",
            "stage": PipelineStage.CHART_GENERATION,
            "trace": self._trace(state, "chart_generator"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="chart_generator",
                title="Chart generation",
                summary=f"Generated Vega-Lite specification; technical attempt {technical_attempt}, semantic attempt {semantic_attempt}",
                inputs=[selected.chart_family if selected else "", f"semantic_feedback={len(semantic_feedback_items)}"],
                outputs=[str(result.spec_json.get("mark", ""))],
                details={"artifact": list(artifact_paths.values())[-1], "spec_json": result.spec_json},
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
        artifact_paths = self._save_into(artifact_paths, state["run_id"], f"spec_validation_attempt_{attempt_number:03d}", payload)
        artifact_paths = self._save_text_into(
            artifact_paths,
            state["run_id"],
            f"spec_validation_attempt_{attempt_number:03d}_report",
            self._validation_attempt_report(attempt_number, payload),
        )
        if validation_result.is_valid:
            artifact_paths = self._save_into(artifact_paths, state["run_id"], "spec_validation", validation_result.model_dump())

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
                    "artifact": artifact_paths.get("spec_validation") or artifact_paths.get(f"spec_validation_attempt_{attempt_number:03d}"),
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
            artifact_paths = self._save_into(artifact_paths, state["run_id"], "technical_retry_summary", summary)
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
            artifact_paths = self._save_into(artifact_paths, state["run_id"], f"technical_retry_decision_{attempt_number:03d}", summary)
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
        artifact_paths = self._save_into(artifact_paths, state["run_id"], "technical_retry_summary", summary)
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
                outputs=[str(result.empty_chart_signal)],
                details={"artifact": artifact_paths["empty_chart_check"], **result.model_dump()},
            ),
        }

    @traceable(name="virage.spec_score")
    def spec_score_node(self, state: PipelineState) -> dict:
        if not self.runtime.settings.enable_spec_score:
            return {}
        ground_truth = state.get("user_context", {}).get("ground_truth_spec")
        if not isinstance(ground_truth, dict):
            return {}
        result = self.spec_score.invoke(state["spec_validation"], ground_truth)
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
        return {
            "semantic_status": "enabled",
            "semantic_attempt_number": max(1, int(state.get("semantic_attempt_number", 1) or 1)),
            "stage": PipelineStage.EVALUATION,
            "trace": self._trace(state, "semantic_loop_gate_enabled"),
            "step_logs": self._append_log(
                state,
                stage="semantic_loop_gate",
                title="Semantic VLM loop",
                summary="enabled",
                outputs=["enabled"],
            ),
        }

    @traceable(name="virage.vlm_chart_description")
    def vlm_chart_description_node(self, state: PipelineState) -> dict:
        plot_image = PlotImageArtifact(**state["plot_image"])
        result = self.vlm_chart_description.invoke(plot_image, runtime=self.runtime)
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        artifact_paths = self._save(state, f"semantic_attempt_{attempt_number:03d}_vlm_chart_description", result.model_dump())
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
                details={"artifact": artifact_paths[f"semantic_attempt_{attempt_number:03d}_vlm_chart_description"], **result.model_dump()},
            ),
        }

    @traceable(name="virage.chart_fact_summary")
    def chart_fact_summary_node(self, state: PipelineState) -> dict:
        result = self.chart_fact_summary.invoke(state["vlm_chart_description"], runtime=self.runtime)
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        artifact_paths = self._save(state, f"semantic_attempt_{attempt_number:03d}_chart_fact_summary", result.model_dump())
        history = [*state.get("semantic_chart_fact_history", []), result.model_dump()]
        return {
            "chart_fact_summary": result,
            "semantic_chart_fact_history": history,
            "stage": PipelineStage.FACT_EXTRACTION,
            "trace": self._trace(state, "chart_fact_summary"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="chart_fact_summary",
                title="Chart fact summary",
                summary=f"{len(result.facts)} facts",
                inputs=["png-only VLM description"],
                outputs=result.facts[:3],
                details={"artifact": artifact_paths[f"semantic_attempt_{attempt_number:03d}_chart_fact_summary"], **result.model_dump()},
            ),
        }

    @traceable(name="virage.chart_answer_judge")
    def chart_answer_judge_node(self, state: PipelineState) -> dict:
        result = self.chart_answer_judge.invoke(
            query=state["query"],
            chart_facts=state["chart_fact_summary"],
            runtime=self.runtime,
            request_analysis=state.get("request_analysis"),
        )
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        artifact_paths = self._save(state, f"semantic_attempt_{attempt_number:03d}_answer_judge", result.model_dump())
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
                details={"artifact": artifact_paths[f"semantic_attempt_{attempt_number:03d}_answer_judge"], **result.model_dump()},
            ),
        }

    @traceable(name="virage.semantic_decision")
    def semantic_decision_node(self, state: PipelineState) -> dict:
        judge = state["chart_answer_judge"]
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        max_attempts = max(1, int(self.runtime.settings.semantic_feedback_max_attempts))
        min_confidence = float(self.runtime.settings.semantic_feedback_min_accept_confidence)
        accepted = bool(judge.answers_user_query and judge.confidence >= min_confidence and judge.retry_recommendation == "accept")
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
            artifact_paths = self._save_into(artifact_paths, state["run_id"], "semantic_feedback_loop_summary", summary.model_dump())
            summary.summary_artifact_path = artifact_paths["semantic_feedback_loop_summary"]
            return {
                "semantic_status": "accept",
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

        feedback_text = judge.feedback_for_next_generation or "\n".join(judge.improvement_comments or judge.missing_requirements)
        feedback_items = [*state.get("semantic_feedback_items", [])]
        if feedback_text:
            feedback_items.append(feedback_text)

        if attempt_number < max_attempts:
            summary = {
                "status": "retry",
                "attempt_number": attempt_number,
                "next_semantic_attempt": attempt_number + 1,
                "max_attempts": max_attempts,
                "confidence": judge.confidence,
                "missing_requirements": judge.missing_requirements,
                "improvement_comments": judge.improvement_comments,
                "feedback_for_next_generation": feedback_text,
            }
            artifact_paths = self._save_into(artifact_paths, state["run_id"], f"semantic_retry_decision_{attempt_number:03d}", summary)
            return {
                "semantic_status": "retry",
                "semantic_attempt_number": attempt_number + 1,
                "technical_attempt_number": 1,
                "technical_retry_feedback": {},
                "semantic_feedback_items": feedback_items,
                "semantic_retry_feedback": feedback_text,
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
            final_feedback_example_path = artifact_paths.get(f"semantic_attempt_{attempt_number:03d}_feedback_example", "")
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
        artifact_paths = self._save_into(artifact_paths, state["run_id"], "semantic_feedback_loop_summary", summary.model_dump())
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
        artifact_paths = self._save(state, f"semantic_attempt_{attempt_number:03d}_feedback_example", example.model_dump())
        corpus_path = ""
        if self.runtime.settings.semantic_feedback_save_rejected_specs:
            corpus_path = self.feedback_corpus_writer.append_to_corpus(example, self.runtime)
        feedback_block = example.feedback_for_next_generation
        if feedback_block:
            artifact_paths = self._save_text_into(artifact_paths, state["run_id"], f"semantic_attempt_{attempt_number:03d}_feedback_prompt_block", feedback_block)
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
                details={"artifact": artifact_paths[f"semantic_attempt_{attempt_number:03d}_feedback_example"], "corpus_path": corpus_path},
            ),
        }

    @traceable(name="virage.vlm_analysis")
    def vlm_analysis_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        plot_image = PlotImageArtifact(**state["plot_image"])
        result = self.vlm_analysis.invoke(plot_image, state["analysis_rubric"], runtime=self.runtime)
        artifact_paths = self._save(state, "vlm_analysis", result.model_dump())
        return {
            "vlm_analysis": result,
            "stage": PipelineStage.VLM_ANALYSIS,
            "trace": self._trace(state, "vlm_analysis"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="vlm_analysis",
                title="Visual analysis",
                summary=f"{len(result.visual_observations)} observations",
                inputs=[state["plot_image"]["image_path"]],
                outputs=result.visual_observations[:3],
                details=self._stage_details(before) | {"artifact": artifact_paths["vlm_analysis"],
                                                       **result.model_dump()},
            ),
        }

    @traceable(name="virage.fact_extractor")
    def fact_extractor_node(self, state: PipelineState) -> dict:
        result = self.fact_extractor.invoke(state["vlm_analysis"], runtime=self.runtime)
        artifact_paths = self._save(state, "visual_facts", result.model_dump())
        return {
            "visual_facts": result,
            "stage": PipelineStage.FACT_EXTRACTION,
            "trace": self._trace(state, "fact_extractor"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="fact_extractor",
                title="Fact extraction",
                summary=f"{len(result.visual_facts)} facts",
                inputs=["visual observations"],
                outputs=[fact.name for fact in result.visual_facts[:3]],
                details={"artifact": artifact_paths["visual_facts"], **result.model_dump()},
            ),
        }

    @traceable(name="virage.reasoner")
    def reasoner_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.reasoner.invoke(state["visual_facts"], state["analysis_rubric"], runtime=self.runtime)
        artifact_paths = self._save(state, "insight_reasoning", result.model_dump())
        return {
            "insight_reasoning": result,
            "stage": PipelineStage.REASONING,
            "trace": self._trace(state, "reasoner"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="reasoner",
                title="Reasoning",
                summary=f"{len(result.insight_candidates)} insight candidates",
                inputs=["visual facts"],
                outputs=result.reasoning_chain[:3],
                details=self._stage_details(before) | {"artifact": artifact_paths["insight_reasoning"],
                                                       **result.model_dump()},
            ),
        }

    @traceable(name="virage.insights")
    def insights_node(self, state: PipelineState) -> dict:
        result = self.insights.invoke(state["insight_reasoning"])
        artifact_paths = self._save(state, "insights", result.model_dump())
        return {
            "insights": result,
            "stage": PipelineStage.INSIGHTS,
            "trace": self._trace(state, "insights"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="insights",
                title="Final insights",
                summary=f"{len(result.final_insights)} insights",
                inputs=["insight candidates"],
                outputs=result.final_insights[:3],
                details={"artifact": artifact_paths["insights"], **result.model_dump()},
            ),
        }

    @traceable(name="virage.vision_score")
    def vision_score_node(self, state: PipelineState) -> dict:
        if not self.runtime.settings.enable_vision_score:
            return {}
        before = len(self.runtime.model_call_logs)
        plot_image = PlotImageArtifact(**state["plot_image"])
        result = self.vision_score.invoke(plot_image, runtime=self.runtime,
                                          query_understanding=state.get("query_understanding"))
        artifact_paths = self._save(state, "visual_quality_metric", result.model_dump())
        return {
            "visual_quality_metric": result,
            "stage": PipelineStage.EVALUATION,
            "trace": self._trace(state, "vision_score"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="vision_score",
                title="Visual metric",
                summary=f"score={result.score:.3f}",
                inputs=[state["plot_image"]["image_path"]],
                outputs=result.details[:3],
                details=self._stage_details(before) | {"artifact": artifact_paths["visual_quality_metric"],
                                                       **result.model_dump()},
            ),
        }

    @traceable(name="virage.evaluation_summary")
    def evaluation_summary_node(self, state: PipelineState) -> dict:
        if not self.runtime.settings.enable_evaluation_summary:
            return {}
        result = self.evaluation_summary.invoke(
            state.get("structural_spec_metric"),
            state.get("visual_quality_metric"),
            state["empty_chart_check"],
            state.get("insights"),
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
