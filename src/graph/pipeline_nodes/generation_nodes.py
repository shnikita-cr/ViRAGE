from __future__ import annotations

from .common import *  # noqa: F401,F403


class GenerationPipelineNodesMixin:
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
            visual_judge_requirements=state.get("visual_judge_requirements"),
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
