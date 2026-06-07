from __future__ import annotations

from src.application.runtime.state import PipelineState
from src.domain.common.enums import PipelineStage
from src.domain.models import PlotImageArtifact, SemanticFeedbackLoopSummary
from src.graph.pipeline_nodes.shared.common import (
    _actionable_semantic_feedback,
    _semantic_feedback_text,
    _semantic_retry_reasons,
)
from src.observability import traceable
from src.services.visual_feedback import VisualChartJudgeAdapters


class VisualFeedbackPipelineNodesMixin:
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

    @traceable(name="virage.visual_chart_judge")
    def visual_chart_judge_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        attempt_number = max(1, int(state.get("semantic_attempt_number", 1) or 1))
        plot_image = PlotImageArtifact(**state["plot_image"])
        result = self.visual_chart_judge.invoke(
            query=state["query"],
            plot_image=plot_image,
            runtime=self.runtime,
            request_analysis=state.get("query_request_analysis"),
            visual_judge_requirements=state.get("visual_judge_requirements"),
        )
        vlm_description = VisualChartJudgeAdapters.to_vlm_description(result)
        chart_facts = VisualChartJudgeAdapters.to_fact_summary(result)
        answer_judge = VisualChartJudgeAdapters.to_answer_judge(result)
        retry_reasons = _semantic_retry_reasons(answer_judge)
        used_fields = []
        if state.get("query_request_analysis") is not None:
            used_fields = list(state["query_request_analysis"].selected_fields)
        chart_analysis = VisualChartJudgeAdapters.to_chart_analysis_record(
            query=state["query"],
            result=result,
            used_fields=used_fields,
        )
        revision = VisualChartJudgeAdapters.to_revision_record(
            attempt_number=attempt_number,
            query=state["query"],
            result=result,
            vega_spec=state["vega_spec"],
            rendered_png_path=state["plot_image"].get("image_path", ""),
            retry_reasons=retry_reasons,
        )
        semantic_payload = {
            "visual_chart_judge": result.model_dump(),
            "chart_analysis": chart_analysis.model_dump(),
            "chart_feedback": answer_judge.model_dump(),
            "chart_revision_record": revision.model_dump(),
        }
        artifact_paths = self._save(state, "visual_chart_judge", semantic_payload)
        history = [*state.get("semantic_chart_fact_history", []), chart_facts.model_dump()]
        return {
            "visual_chart_judge": result,
            "vlm_chart_description": vlm_description,
            "chart_fact_summary": chart_facts,
            "semantic_chart_fact_history": history,
            "chart_answer_judge": answer_judge,
            "chart_analysis": chart_analysis,
            "chart_revision_record": revision,
            "semantic_retry_reasons": retry_reasons,
            "stage": PipelineStage.VERIFICATION,
            "trace": self._trace(state, "visual_chart_judge"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="visual_chart_judge",
                title="PNG-only visual chart judge",
                summary=f"answers={result.answers_user_query}; recommendation={result.retry_recommendation}; confidence={result.confidence:.3f}",
                inputs=[state["plot_image"].get("image_path", ""), "user_query", "visual_judge_requirements"],
                outputs=[result.retry_recommendation, *retry_reasons[:2]],
                details=self._stage_details(before) | {
                    "artifact": artifact_paths["visual_chart_judge"],
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
            request_analysis=state.get("query_request_analysis"),
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
        artifact_paths = dict(state.get("artifact_paths", {}))

        if self._semantic_judge_accepted(judge, actionable_feedback, min_confidence=min_confidence):
            return self._semantic_accept_state(
                state=state,
                judge=judge,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                artifact_paths=artifact_paths,
            )

        feedback_text = self._semantic_feedback_text(judge, retry_reasons)
        feedback_items = self._semantic_feedback_items(state, feedback_text)
        if retry_reasons and attempt_number < max_attempts:
            return self._semantic_retry_state(
                state=state,
                judge=judge,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                actionable_feedback=actionable_feedback,
                retry_reasons=retry_reasons,
                feedback_text=feedback_text,
                feedback_items=feedback_items,
                artifact_paths=artifact_paths,
            )

        return self._semantic_failed_state(
            state=state,
            judge=judge,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
            artifact_paths=artifact_paths,
        )

    @staticmethod
    def _semantic_judge_accepted(judge, actionable_feedback: list[str], *, min_confidence: float) -> bool:
        return bool(
            judge.answers_user_query
            and judge.confidence >= min_confidence
            and (judge.retry_recommendation == "accept" or not actionable_feedback)
        )

    @staticmethod
    def _semantic_feedback_text(judge, retry_reasons: list[str]) -> str:
        feedback_text = _semantic_feedback_text(judge)
        if not feedback_text and retry_reasons:
            feedback_text = "\n".join(retry_reasons)
        return feedback_text

    @staticmethod
    def _semantic_feedback_items(state: PipelineState, feedback_text: str) -> list[str]:
        feedback_items = [*state.get("semantic_feedback_items", [])]
        if feedback_text:
            feedback_items.append(feedback_text)
        return feedback_items

    def _semantic_accept_state(
            self,
            *,
            state: PipelineState,
            judge,
            attempt_number: int,
            max_attempts: int,
            artifact_paths: dict[str, str],
    ) -> dict:
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

    def _semantic_retry_state(
            self,
            *,
            state: PipelineState,
            judge,
            attempt_number: int,
            max_attempts: int,
            actionable_feedback: list[str],
            retry_reasons: list[str],
            feedback_text: str,
            feedback_items: list[str],
            artifact_paths: dict[str, str],
    ) -> dict:
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
        artifact_paths = self._save_into(artifact_paths, state["run_id"], "semantic_decision", summary)
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

    def _semantic_failed_state(
            self,
            *,
            state: PipelineState,
            judge,
            attempt_number: int,
            max_attempts: int,
            artifact_paths: dict[str, str],
    ) -> dict:
        final_examples = list(state.get("visual_feedback_examples", []))
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
                request_analysis=state.get("query_request_analysis"),
                data_profile=state.get("data_profile"),
            )
            artifact_paths = self._save_into(
                artifact_paths,
                state["run_id"],
                f"semantic_attempt_{attempt_number:03d}_feedback_example",
                final_example.model_dump(),
            )
            final_examples.append(final_example)
            if self.runtime.settings.semantic_feedback_save_rejected_specs:
                final_corpus_path = self.feedback_corpus_writer.append_to_corpus(final_example, self.runtime)
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as exc:
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
            request_analysis=state.get("query_request_analysis"),
            data_profile=state.get("data_profile"),
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

    def semantic_chart_judge_node(self, state: PipelineState) -> dict:
        return self.visual_chart_judge_node(state)
