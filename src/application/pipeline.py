from __future__ import annotations

import traceback
from typing import Callable

from src.application.bootstrap import bootstrap_project_environment
from src.application.contracts import PipelineRequest, PipelineResult
from src.application.project_config import ProjectConfig
from src.application.settings import ViRAGESettings
from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.domain.models import ModelCallLog, StepLog
from src.graph.builder import build_pipeline_graph
from src.infrastructure.runtime import RuntimeContext
from src.llm.factory import build_chat_model
from src.llm.healthcheck import check_required_models, raise_for_failed_health_checks
from src.observability import traceable


def _classify_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "health check" in text or "model" in text and "not found" in text:
        return "model_unavailable"
    if "timeout" in text or "timed out" in text:
        return "model_timeout"
    if "parse" in text or "json" in text:
        return "parse_failed"
    if "503" in text or "overloaded" in text:
        return "model_unavailable"
    return "failed"


def _stage_name(value: object) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", str(value))


class ViRAGEPipeline:
    def __init__(
            self,
            settings: ViRAGESettings | None = None,
            reasoning_llm: object | None = None,
            spec_llm: object | None = None,
            vlm: object | None = None,
            vision_judge_llm: object | None = None,
    ) -> None:
        bootstrap_project_environment()
        self.settings = settings or ViRAGESettings()
        self.runtime = RuntimeContext(
            settings=self.settings,
            reasoning_llm=reasoning_llm,
            spec_llm=spec_llm,
            vlm=vlm,
            vision_judge_llm=vision_judge_llm,
        )
        self.graph = build_pipeline_graph(self.runtime)

    @classmethod
    def from_project_config(cls, config: ProjectConfig) -> 'ViRAGEPipeline':
        settings = config.settings.model_copy(deep=True)
        if config.mode == 'streamlit' and not config.streamlit.compute_metrics:
            settings.enable_spec_score = False
            settings.enable_vision_score = False
            settings.enable_evaluation_summary = False
        settings.streamlit_compute_metrics = config.streamlit.compute_metrics
        settings.streamlit_show_step_logs = config.streamlit.show_step_logs
        models = {
            "reasoning": build_chat_model(config.reasoning_model),
            "spec": build_chat_model(config.spec_model),
            "vlm": build_chat_model(config.vlm_model),
            "vision_judge": build_chat_model(config.vision_judge_model),
        }
        if bool(getattr(settings, "model_health_check_enabled", False)):
            required_roles = set(getattr(settings, "model_health_check_required_roles", []) or [])
            active_models = {role: model for role, model in models.items() if
                             not required_roles or role in required_roles}
            results = check_required_models(
                active_models,
                timeout_seconds=float(getattr(settings, "model_health_check_timeout_seconds", 10.0)),
            )
            raise_for_failed_health_checks(results)
        return cls(
            settings=settings,
            reasoning_llm=models["reasoning"],
            spec_llm=models["spec"],
            vlm=models["vlm"],
            vision_judge_llm=models["vision_judge"],
        )

    def _save_input_artifacts(self, request: PipelineRequest) -> dict[str, str]:
        query_path = self.runtime.save_text_artifact(
            "input/query.txt",
            request.query,
            run_id=request.run_id,
        )
        context_path = self.runtime.save_json_artifact(
            "input/context.json",
            request.user_context or {},
            run_id=request.run_id,
        )
        return {
            "input_query": query_path,
            "input_context": context_path,
        }

    @traceable(name='virage.pipeline.invoke')
    def invoke(
            self,
            request: PipelineRequest,
            *,
            step_callback: Callable[[StepLog], None] | None = None,
            model_call_callback: Callable[[ModelCallLog], None] | None = None,
    ) -> PipelineResult:
        self.runtime.current_run_id = request.run_id
        self.runtime.reset_model_logs()
        self.runtime.reset_stage_execution_logs()
        self.runtime.reset_artifact_indices(run_id=request.run_id)
        self.runtime.step_callback = step_callback
        self.runtime.model_call_callback = model_call_callback
        self.runtime.ensure_run_dir(request.run_id)
        input_artifact_paths = self._save_input_artifacts(request)
        self.runtime.save_run_status(
            run_id=request.run_id,
            status="running",
            final_stage="initialized",
            extra={"query": request.query, "data_path": request.data_path},
        )
        initial_state: PipelineState = {
            'run_id': request.run_id,
            'query': request.query,
            'data_path': request.data_path,
            'user_context': request.user_context,
            'stage': PipelineStage.INITIALIZED,
            'trace': [],
            'errors': [],
            'step_logs': [],
            'model_call_logs': [],
            'stage_execution_logs': [],
            'artifact_paths': input_artifact_paths,
        }
        try:
            final_state: PipelineState = self.graph.invoke(
                initial_state,
                config={"recursion_limit": int(self.settings.graph_recursion_limit)},
            )
            final_state['stage'] = PipelineStage.COMPLETED
        except Exception as exc:
            tb = traceback.format_exc()
            error_type = _classify_error(exc)
            self.runtime.save_text_artifact('errors/fatal_error.txt', tb, run_id=request.run_id, numbered=True)
            self.runtime.save_run_status(
                run_id=request.run_id,
                status="failed",
                final_stage="exception",
                semantic_status=None,
                error_type=error_type,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.runtime.save_model_log_artifacts(run_id=request.run_id)
            raise
        finally:
            self.runtime.step_callback = None
            self.runtime.model_call_callback = None
        final_state['model_call_logs'] = list(self.runtime.model_call_logs)
        final_state['stage_execution_logs'] = list(self.runtime.stage_execution_logs)
        final_state['token_usage_summary'] = self.runtime.token_usage_summary()
        self.runtime.save_model_log_artifacts(run_id=request.run_id)
        self.runtime.save_run_status(
            run_id=request.run_id,
            status="completed",
            final_stage=_stage_name(final_state.get('stage')),
            semantic_status=str(final_state.get('semantic_status') or ""),
            extra={
                "query": request.query,
                "data_path": request.data_path,
                "has_plot": bool(final_state.get("plot_image")),
                "has_evaluation_summary": bool(final_state.get("evaluation_summary")),
                "has_token_summary": True,
            },
        )
        return PipelineResult(
            run_id=final_state['run_id'],
            query=final_state['query'],
            data_path=final_state['data_path'],
            query_request_analysis=final_state.get('query_request_analysis'),
            analysis_rubric=final_state.get('analysis_rubric'),
            data_profile=final_state.get('data_profile'),
            data_preparation=final_state.get('data_preparation'),
            visrag=final_state.get('visrag'),
            candidate_spec_set=final_state.get('candidate_spec_set'),
            vega_spec=final_state.get('vega_spec'),
            spec_validation=final_state.get('spec_validation'),
            plot_rendering=final_state.get('plot_rendering'),
            scenegraph_check=final_state.get('scenegraph_check'),
            empty_chart_check=final_state.get('empty_chart_check'),
            plot_image=final_state.get('plot_image'),
            vlm_chart_description=final_state.get('vlm_chart_description'),
            chart_fact_summary=final_state.get('chart_fact_summary'),
            chart_answer_judge=final_state.get('chart_answer_judge'),
            visual_feedback_examples=final_state.get('visual_feedback_examples', []),
            semantic_feedback_loop_summary=final_state.get('semantic_feedback_loop_summary'),
            vlm_analysis=final_state.get('vlm_analysis'),
            structural_spec_metric=final_state.get('structural_spec_metric'),
            evaluation_summary=final_state.get('evaluation_summary'),
            step_logs=final_state.get('step_logs', []),
            stage_execution_logs=final_state.get('stage_execution_logs', []),
            model_call_logs=final_state.get('model_call_logs', []),
            token_usage_summary=final_state.get('token_usage_summary', self.runtime.token_usage_summary()),
            artifact_paths=final_state.get('artifact_paths', {}),
        )
