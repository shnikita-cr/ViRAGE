from __future__ import annotations

from src.application.contracts import PipelineResult
from src.application.runtime.state import PipelineState
from src.infrastructure.runtime import RuntimeContext


class PipelineResultFactory:
    """Build a public PipelineResult from the internal mutable graph state."""

    RESULT_STATE_FIELDS = (
        "query_request_analysis",
        "analysis_rubric",
        "data_profile",
        "data_preparation",
        "visrag",
        "vega_spec",
        "spec_validation",
        "plot_rendering",
        "scenegraph_check",
        "empty_chart_check",
        "plot_image",
        "vlm_chart_description",
        "chart_fact_summary",
        "chart_answer_judge",
        "visual_chart_judge",
        "visual_feedback_examples",
        "semantic_feedback_loop_summary",
        "vlm_analysis",
        "insights",
        "structural_spec_metric",
        "evaluation_summary",
        "step_logs",
        "stage_execution_logs",
        "model_call_logs",
        "token_usage_summary",
        "artifact_paths",
    )

    @classmethod
    def from_state(cls, final_state: PipelineState, runtime: RuntimeContext) -> PipelineResult:
        return PipelineResult(
            run_id=final_state["run_id"],
            query=final_state["query"],
            data_path=final_state["data_path"],
            query_request_analysis=final_state.get("query_request_analysis"),
            analysis_rubric=final_state.get("analysis_rubric"),
            data_profile=final_state.get("data_profile"),
            data_preparation=final_state.get("data_preparation"),
            visrag=final_state.get("visrag"),
            vega_spec=final_state.get("vega_spec"),
            spec_validation=final_state.get("spec_validation"),
            plot_rendering=final_state.get("plot_rendering"),
            scenegraph_check=final_state.get("scenegraph_check"),
            empty_chart_check=final_state.get("empty_chart_check"),
            plot_image=final_state.get("plot_image"),
            vlm_chart_description=final_state.get("vlm_chart_description"),
            chart_fact_summary=final_state.get("chart_fact_summary"),
            chart_answer_judge=final_state.get("chart_answer_judge"),
            visual_chart_judge=final_state.get("visual_chart_judge"),
            visual_feedback_examples=final_state.get("visual_feedback_examples", []),
            semantic_feedback_loop_summary=final_state.get("semantic_feedback_loop_summary"),
            vlm_analysis=final_state.get("vlm_analysis"),
            insights=final_state.get("insights"),
            structural_spec_metric=final_state.get("structural_spec_metric"),
            evaluation_summary=final_state.get("evaluation_summary"),
            step_logs=final_state.get("step_logs", []),
            stage_execution_logs=final_state.get("stage_execution_logs", []),
            model_call_logs=final_state.get("model_call_logs", []),
            token_usage_summary=final_state.get("token_usage_summary", runtime.token_usage_summary()),
            artifact_paths=final_state.get("artifact_paths", {}),
        )
