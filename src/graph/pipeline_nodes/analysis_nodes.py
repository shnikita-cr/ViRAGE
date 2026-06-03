from __future__ import annotations

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.domain.models import InsightsResult, PlotImageArtifact, VLMAnalysisResult
from src.observability import traceable


class AnalysisPipelineNodesMixin:
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
