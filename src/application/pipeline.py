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
from src.observability import traceable


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
        return cls(
            settings=settings,
            reasoning_llm=build_chat_model(config.reasoning_model),
            spec_llm=build_chat_model(config.spec_model),
            vlm=build_chat_model(config.vlm_model),
            vision_judge_llm=build_chat_model(config.vision_judge_model),
        )

    @traceable(name='virage.pipeline.invoke')
    def invoke(
            self,
            request: PipelineRequest,
            *,
            step_callback: Callable[[StepLog], None] | None = None,
            model_call_callback: Callable[[ModelCallLog], None] | None = None,
    ) -> PipelineResult:
        self.runtime.reset_model_logs()
        self.runtime.current_run_id = request.run_id
        self.runtime.step_callback = step_callback
        self.runtime.model_call_callback = model_call_callback
        self.runtime.ensure_run_dir(request.run_id)
        self.runtime.save_text_artifact('input/query.txt', request.query, run_id=request.run_id)
        self.runtime.save_json_artifact('input/user_context.json', request.user_context, run_id=request.run_id)
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
            'artifact_paths': {},
        }
        try:
            final_state: PipelineState = self.graph.invoke(initial_state)
            final_state['stage'] = PipelineStage.COMPLETED
        except Exception as exc:
            tb = traceback.format_exc()
            self.runtime.save_text_artifact('errors/fatal_error.txt', tb, run_id=request.run_id)
            raise
        finally:
            self.runtime.step_callback = None
            self.runtime.model_call_callback = None
        final_state['model_call_logs'] = list(self.runtime.model_call_logs)
        final_state['token_usage_summary'] = self.runtime.token_usage_summary()
        self.runtime.save_json_artifact('artifacts/model_call_logs.json',
                                        [item.model_dump() for item in self.runtime.model_call_logs],
                                        run_id=request.run_id)
        self.runtime.save_json_artifact('artifacts/token_usage_summary.json',
                                        final_state['token_usage_summary'].model_dump(), run_id=request.run_id)
        return PipelineResult(
            run_id=final_state['run_id'],
            query=final_state['query'],
            data_path=final_state['data_path'],
            query_understanding=final_state.get('query_understanding'),
            query_intent_bundle=final_state.get('query_intent_bundle'),
            request_analysis=final_state.get('request_analysis'),
            planning=final_state.get('planning'),
            execution_policy=final_state.get('execution_policy'),
            validation_policy=final_state.get('validation_policy'),
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
            vlm_analysis=final_state.get('vlm_analysis'),
            visual_facts=final_state.get('visual_facts'),
            insight_reasoning=final_state.get('insight_reasoning'),
            insight_verification=final_state.get('insight_verification'),
            insights=final_state.get('insights'),
            structural_spec_metric=final_state.get('structural_spec_metric'),
            visual_quality_metric=final_state.get('visual_quality_metric'),
            evaluation_summary=final_state.get('evaluation_summary'),
            step_logs=final_state.get('step_logs', []),
            model_call_logs=final_state.get('model_call_logs', []),
            token_usage_summary=final_state.get('token_usage_summary', self.runtime.token_usage_summary()),
            artifact_paths=final_state.get('artifact_paths', {}),
        )
