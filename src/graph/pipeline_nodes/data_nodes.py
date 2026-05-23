from __future__ import annotations

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.graph.pipeline_nodes.common import _data_profile_artifact_payload
from src.observability import traceable

class DataPipelineNodesMixin:
    @traceable(name="virage.data_profiler")
    def data_profiler_node(self, state: PipelineState) -> dict:
        result = self.data_profiler.invoke(state["data_path"], runtime=self.runtime)
        artifact_paths = self._save(state, "data_profile", _data_profile_artifact_payload(result, self.runtime))
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
                outputs=[", ".join(column.name for column in result.measure_columns()[:3]), ", ".join(column.name for column in result.temporal_columns()[:3])],
                details={"artifact": artifact_paths["data_profile"]},
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
        )
        artifact_paths = self._save(state, "query_request_analysis", result.model_dump())
        return {
            "query_request_analysis": result,
            "visual_judge_requirements": result.visual_judge_requirements,
            "stage": PipelineStage.QUERY_REQUEST_ANALYSIS,
            "trace": self._trace(state, "query_request_analysis"),
            "artifact_paths": artifact_paths,
            "step_logs": self._append_log(
                state,
                stage="query_request_analysis",
                title="Query and request analysis",
                summary=f"{result.normalized_query}; fields={', '.join(result.selected_fields[:5])}",
                inputs=[state["query"], "data_profile"],
                outputs=[result.analysis_task or "unknown", *result.selected_fields[:4]],
                details=self._stage_details(before) | {"artifact": artifact_paths["query_request_analysis"]},
            ),
        }

    @traceable(name="virage.data_preparation")
    def data_preparation_node(self, state: PipelineState) -> dict:
        result = self.data_preparation.invoke(
            state["data_path"],
            state["data_profile"],
            state["query_request_analysis"],
            state["run_id"],
            runtime=self.runtime,
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
                details={
                    "artifact": artifact_paths["data_preparation"],
                    "prepared_path": result.output_path,
                },
            ),
        }
