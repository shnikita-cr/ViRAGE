from __future__ import annotations

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.graph.pipeline_nodes.common import (
    _manual_feedback_items,
    _merge_unique_texts,
    _vega_spec_artifact_payload,
)
from src.observability import traceable
from src.visrag_core.task_context import task_context_from_user_context

class GenerationPipelineNodesMixin:
    @traceable(name="virage.visrag")
    def visrag_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        task_context = task_context_from_user_context(state.get("user_context", {}))
        result = self.visrag.invoke(
            state["query_request_analysis"],
            state["data_profile"],
            runtime=self.runtime,
            task_context=task_context,
        )
        artifact_paths = self._save(state, "visrag", result.model_dump())
        guidance_summary = result.generation_guidance.prompt_text.splitlines()[0] if result.generation_guidance.prompt_text else "no guidance"
        retrieved_types = result.diagnostics.retrieved_count_by_type
        return {
            "visrag": result,
            "analysis_rubric": self._analysis_rubric(state),
            "stage": PipelineStage.VISRAG,
            "trace": self._trace(state, "visrag"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="visrag",
                title="Rule guidance retrieval",
                summary=guidance_summary,
                inputs=[state["query_request_analysis"].normalized_query, f"task_type={task_context.get('task_type', 'single_chart')}"] ,
                outputs=[f"{key}:{value}" for key, value in sorted(retrieved_types.items())],
                details=self._stage_details(before) | {"artifact": artifact_paths["visrag"], "task_context": task_context},
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
            runtime=self.runtime,
            query=state["query"],
            data_profile=state.get("data_profile"),
            query_request_analysis=state.get("query_request_analysis"),
            visrag=state.get("visrag"),
            generation_attempt_number=technical_attempt,
            max_generation_attempts=max_generation_attempts,
            previous_validation_errors=list(technical_feedback.get("validation_errors") or []),
            previous_repair_hints=list(technical_feedback.get("repair_hints") or []),
            previous_invalid_spec=technical_feedback.get("invalid_spec"),
            previous_semantic_feedback=semantic_feedback_items,
            previous_chart_facts=semantic_chart_fact_history,
            visual_judge_requirements=state.get("visual_judge_requirements"),
        )
        artifact_paths = self._save(state, "vega_spec", _vega_spec_artifact_payload(result))
        guidance_present = bool(state.get("visrag") and state["visrag"].generation_guidance.has_guidance)
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
                inputs=[f"visrag_guidance={guidance_present}", f"semantic_feedback={len(semantic_feedback_items)}"],
                outputs=[str((result.spec_without_runtime_data or result.spec_json).get("mark", ""))],
                details={"artifact": artifact_paths["vega_spec"], "spec": _vega_spec_artifact_payload(result)["spec"]},
            ),
        }
