from __future__ import annotations

from .common import *  # noqa: F401,F403


class DataPipelineNodesMixin:
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
            "visual_judge_requirements": result.visual_judge_requirements,
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
