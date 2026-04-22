from __future__ import annotations

from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.domain.models import PlotImageArtifact, StepLog
from src.infrastructure.runtime import RuntimeContext
from src.observability import traceable
from src.services.chart_generator import ChartGeneratorService
from src.services.data_preparation import DataPreparationService
from src.services.data_profiler import DataProfilerService
from src.services.empty_chart_check import EmptyChartCheckService
from src.services.evaluation_summary import EvaluationSummaryService
from src.services.fact_extractor import FactExtractorService
from src.services.insights import InsightsService
from src.services.planning import PlanningService
from src.services.query_understanding import QueryUnderstandingService
from src.services.reasoner import ReasonerService
from src.services.request_analyzer import RequestAnalyzerService
from src.services.scenegraph_check import ScenegraphCheckService
from src.services.spec_score import SpecScoreService
from src.services.spec_validator import SpecValidatorService
from src.services.vegalite_plot_drawing import VegaLitePlotDrawingService
from src.services.verifier import VerifierService
from src.services.vision_score import VisionScoreService
from src.services.visrag import VisRAGService
from src.services.vlm_analysis import VLMAnalysisService


class PipelineNodes:
    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.query_understanding = QueryUnderstandingService()
        self.data_profiler = DataProfilerService()
        self.request_analyzer = RequestAnalyzerService()
        self.data_preparation = DataPreparationService()
        self.visrag = VisRAGService()
        self.planning = PlanningService()
        self.chart_generator = ChartGeneratorService()
        self.spec_validator = SpecValidatorService()
        self.vegalite_plot_drawing = VegaLitePlotDrawingService()
        self.scenegraph_check = ScenegraphCheckService()
        self.empty_chart_check = EmptyChartCheckService()
        self.vlm_analysis = VLMAnalysisService()
        self.fact_extractor = FactExtractorService()
        self.reasoner = ReasonerService()
        self.verifier = VerifierService()
        self.insights = InsightsService()
        self.spec_score = SpecScoreService()
        self.vision_score = VisionScoreService()
        self.evaluation_summary = EvaluationSummaryService()

    @staticmethod
    def _trace(state: PipelineState, label: str) -> list[str]:
        return [*state.get('trace', []), label]

    def _stage_details(self, before: int) -> dict:
        logs = self.runtime.model_call_logs[before:]
        return {
            'model_calls': [item.model_dump() for item in logs],
            'stage_token_usage': {
                'prompt_tokens': sum(item.token_usage.prompt_tokens for item in logs),
                'completion_tokens': sum(item.token_usage.completion_tokens for item in logs),
                'total_tokens': sum(item.token_usage.total_tokens for item in logs),
            },
        }

    def _append_log(self, state: PipelineState, *, stage: str, title: str, summary: str, inputs: list[str] | None = None, outputs: list[str] | None = None, details: dict | None = None) -> list[StepLog]:
        return [*state.get('step_logs', []), StepLog(stage=stage, title=title, summary=summary, inputs=inputs or [], outputs=outputs or [], details=details or {})]

    @traceable(name='virage.query_understanding_node')
    def query_understanding_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.query_understanding.invoke(state['query'], state.get('user_context', {}), runtime=self.runtime)
        return {'query_understanding': result, 'query_intent_bundle': result.to_intent_bundle(), 'stage': PipelineStage.QUERY_UNDERSTANDING, 'trace': self._trace(state, 'query_understanding'), 'step_logs': self._append_log(state, stage='query_understanding', title='Query understanding', summary=result.intent, inputs=[state['query']], outputs=[result.task_type or 'unknown', result.analysis_goal or ''], details=self._stage_details(before))}

    @traceable(name='virage.data_profiler_node')
    def data_profiler_node(self, state: PipelineState) -> dict:
        result = self.data_profiler.invoke(state['data_path'], runtime=self.runtime)
        return {'data_profile': result, 'stage': PipelineStage.DATA_PROFILING, 'trace': self._trace(state, 'data_profiler'), 'step_logs': self._append_log(state, stage='data_profiler', title='Data profiling', summary=f'Rows={result.row_count}, Cols={result.col_count}', inputs=[state['data_path']], outputs=[', '.join(result.likely_numeric_columns[:3]), ', '.join(result.likely_time_columns[:3])], details={'field_roles': result.field_roles, 'schema_hints': result.schema_hints})}

    @traceable(name='virage.request_analyzer_node')
    def request_analyzer_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.request_analyzer.invoke(query=state['query'], query_understanding=state['query_understanding'], data_profile=state['data_profile'], runtime=self.runtime)
        return {'request_analysis': result, 'stage': PipelineStage.REQUEST_ANALYSIS, 'trace': self._trace(state, 'request_analyzer'), 'step_logs': self._append_log(state, stage='request_analyzer', title='Request analysis', summary=f"Selected fields: {', '.join(result.selected_fields)}", inputs=[state['query']], outputs=result.grounded_fields[:5], details=self._stage_details(before))}

    @traceable(name='virage.data_preparation_node')
    def data_preparation_node(self, state: PipelineState) -> dict:
        result = self.data_preparation.invoke(state['data_path'], state['data_profile'], state['request_analysis'], state['run_id'], runtime=self.runtime)
        return {'data_preparation': result, 'stage': PipelineStage.DATA_PREPARATION, 'trace': self._trace(state, 'data_preparation'), 'step_logs': self._append_log(state, stage='data_preparation', title='Data preparation', summary=f'Prepared rows={result.row_count}', inputs=[state['data_path']], outputs=result.operations[:5], details={'prepared_path': result.output_path})}

    @traceable(name='virage.visrag_node')
    def visrag_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.visrag.invoke(state['query_understanding'], state['request_analysis'], state['data_profile'], runtime=self.runtime)
        return {'visrag': result, 'candidate_spec_set': result.candidate_spec_set, 'stage': PipelineStage.VISRAG, 'trace': self._trace(state, 'visrag'), 'step_logs': self._append_log(state, stage='visrag', title='Spec retrieval', summary=(result.visualization_plan.title if result.visualization_plan else 'no plan'), inputs=[state['query_understanding'].intent], outputs=[item.chart_family for item in result.recommendations[:3]], details=self._stage_details(before) | {'retrieval_query': result.retrieval_query, 'corpus_status': result.corpus_status})}

    @traceable(name='virage.planning_node')
    def planning_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.planning.invoke(state['query_understanding'], state['request_analysis'], state['data_profile'], state['visrag'], runtime=self.runtime)
        return {'planning': result, 'execution_policy': result.execution_policy, 'validation_policy': result.validation_policy, 'analysis_rubric': result.analysis_rubric, 'stage': PipelineStage.PLANNING, 'trace': self._trace(state, 'planning'), 'step_logs': self._append_log(state, stage='planning', title='Execution planning', summary='Built execution, validation and analysis policies', inputs=[state['query_understanding'].analysis_goal or ''], outputs=[result.execution_policy.retry_strategy if result.execution_policy else '', result.analysis_rubric.output_format if result.analysis_rubric else ''], details=self._stage_details(before))}

    @traceable(name='virage.chart_generator_node')
    def chart_generator_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.chart_generator.invoke(state['data_preparation'], state['candidate_spec_set'], state['execution_policy'], state['validation_policy'], runtime=self.runtime)
        selected = state.get('candidate_spec_set').selected_candidate_spec if state.get('candidate_spec_set') else None
        return {'vega_spec': result, 'stage': PipelineStage.CHART_GENERATION, 'trace': self._trace(state, 'chart_generator'), 'step_logs': self._append_log(state, stage='chart_generator', title='Chart generation', summary='Generated Vega-Lite specification', inputs=[selected.chart_family if selected else ''], outputs=[str(result.spec_json.get('mark', ''))], details=self._stage_details(before) | {'spec_json': result.spec_json})}

    @traceable(name='virage.spec_validator_node')
    def spec_validator_node(self, state: PipelineState) -> dict:
        max_retries = state.get('execution_policy').max_retries if state.get('execution_policy') else 0
        current_spec = state['vega_spec']
        latest = self.spec_validator.invoke(current_spec)
        attempts = 0
        before = len(self.runtime.model_call_logs)
        while not latest.is_valid and attempts < max_retries:
            attempts += 1
            current_spec = self.chart_generator.repair(state['data_preparation'], current_spec, latest.validation_errors, latest.repair_hints, runtime=self.runtime)
            latest = self.spec_validator.invoke(current_spec)
        payload = {'spec_validation': latest, 'stage': PipelineStage.SPEC_VALIDATION, 'trace': self._trace(state, 'spec_validator'), 'step_logs': self._append_log(state, stage='spec_validator', title='Specification validation', summary=('valid' if latest.is_valid else 'invalid'), inputs=['vega_spec'], outputs=latest.validation_errors[:5], details=self._stage_details(before) | {'validated_spec': latest.validated_spec})}
        if current_spec is not state['vega_spec']:
            payload['vega_spec'] = current_spec
        return payload

    @traceable(name='virage.vegalite_plot_drawing_node')
    def vegalite_plot_drawing_node(self, state: PipelineState) -> dict:
        result = self.vegalite_plot_drawing.invoke(state['spec_validation'], state['run_id'], runtime=self.runtime)
        return {'plot_rendering': result, 'plot_image': result.plot_image.model_dump(), 'stage': PipelineStage.PLOT_RENDERING, 'trace': self._trace(state, 'vegalite_plot_drawing'), 'step_logs': self._append_log(state, stage='vegalite_plot_drawing', title='Plot rendering', summary=result.plot_image.image_path, inputs=['validated_spec'], outputs=result.render_notes[:5], details={'rendered_scenegraph': result.rendered_scenegraph})}

    @traceable(name='virage.scenegraph_check_node')
    def scenegraph_check_node(self, state: PipelineState) -> dict:
        result = self.scenegraph_check.invoke(state['plot_rendering'])
        return {'scenegraph_check': result, 'stage': PipelineStage.SCENEGRAPH_CHECK, 'trace': self._trace(state, 'scenegraph_check'), 'step_logs': self._append_log(state, stage='scenegraph_check', title='Scenegraph check', summary='Scenegraph inspected', inputs=['rendered_scenegraph'], outputs=result.notes[:5], details=result.model_dump())}

    @traceable(name='virage.empty_chart_check_node')
    def empty_chart_check_node(self, state: PipelineState) -> dict:
        result = self.empty_chart_check.invoke(state['scenegraph_check'])
        current_spec = state['vega_spec']; current_validation = state['spec_validation']; current_rendering = state['plot_rendering']; current_scenegraph = state['scenegraph_check']; candidate_set = state['candidate_spec_set']
        if result.empty_chart_signal and state['execution_policy'].fallback_enabled and candidate_set and candidate_set.selected_candidate_spec is not None:
            current_index = next((index for index, item in enumerate(candidate_set.candidate_specs) if item.spec_id == candidate_set.selected_candidate_spec.spec_id), 0)
            fallback_index = current_index + 1
            if fallback_index < len(candidate_set.candidate_specs):
                current_spec = self.chart_generator.build_from_candidate(state['data_preparation'], candidate_set, fallback_index, state['execution_policy'], state['validation_policy'], runtime=self.runtime)
                current_validation = self.spec_validator.invoke(current_spec)
                current_rendering = self.vegalite_plot_drawing.invoke(current_validation, state['run_id'], runtime=self.runtime)
                current_scenegraph = self.scenegraph_check.invoke(current_rendering)
                result = self.empty_chart_check.invoke(current_scenegraph)
        payload = {'empty_chart_check': result, 'stage': PipelineStage.EMPTY_CHART_CHECK, 'trace': self._trace(state, 'empty_chart_check'), 'step_logs': self._append_log(state, stage='empty_chart_check', title='Empty chart check', summary=result.empty_chart_status, inputs=['scenegraph_status'], outputs=[str(result.empty_chart_signal)], details=result.model_dump())}
        if current_spec is not state['vega_spec']:
            payload.update({'candidate_spec_set': candidate_set, 'vega_spec': current_spec, 'spec_validation': current_validation, 'plot_rendering': current_rendering, 'plot_image': current_rendering.plot_image.model_dump(), 'scenegraph_check': current_scenegraph})
        return payload

    @traceable(name='virage.spec_score_node')
    def spec_score_node(self, state: PipelineState) -> dict:
        if not self.runtime.settings.enable_spec_score:
            return {}
        result = self.spec_score.invoke(state['spec_validation'])
        return {'structural_spec_metric': result, 'stage': PipelineStage.EVALUATION, 'trace': self._trace(state, 'spec_score'), 'step_logs': self._append_log(state, stage='spec_score', title='Structural metric', summary=f'score={result.score:.3f}', inputs=['validated spec'], outputs=result.details[:3], details=result.model_dump())}

    @traceable(name='virage.vlm_analysis_node')
    def vlm_analysis_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        plot_image = PlotImageArtifact(**state['plot_image'])
        result = self.vlm_analysis.invoke(plot_image, state['analysis_rubric'], runtime=self.runtime)
        return {'vlm_analysis': result, 'stage': PipelineStage.VLM_ANALYSIS, 'trace': self._trace(state, 'vlm_analysis'), 'step_logs': self._append_log(state, stage='vlm_analysis', title='Visual analysis', summary=f'{len(result.visual_observations)} observations', inputs=[state['plot_image']['image_path']], outputs=result.visual_observations[:3], details=self._stage_details(before) | result.model_dump())}

    @traceable(name='virage.fact_extractor_node')
    def fact_extractor_node(self, state: PipelineState) -> dict:
        result = self.fact_extractor.invoke(state['vlm_analysis'], runtime=self.runtime)
        return {'visual_facts': result, 'stage': PipelineStage.FACT_EXTRACTION, 'trace': self._trace(state, 'fact_extractor'), 'step_logs': self._append_log(state, stage='fact_extractor', title='Fact extraction', summary=f'{len(result.visual_facts)} facts', inputs=['visual observations'], outputs=[fact.name for fact in result.visual_facts[:3]], details=result.model_dump())}

    @traceable(name='virage.reasoner_node')
    def reasoner_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.reasoner.invoke(state['visual_facts'], state['analysis_rubric'], runtime=self.runtime)
        return {'insight_reasoning': result, 'stage': PipelineStage.REASONING, 'trace': self._trace(state, 'reasoner'), 'step_logs': self._append_log(state, stage='reasoner', title='Reasoning', summary=f'{len(result.insight_candidates)} insight candidates', inputs=['visual facts'], outputs=result.reasoning_chain[:3], details=self._stage_details(before) | result.model_dump())}

    @traceable(name='virage.verifier_node')
    def verifier_node(self, state: PipelineState) -> dict:
        before = len(self.runtime.model_call_logs)
        result = self.verifier.invoke(state['insight_reasoning'], runtime=self.runtime)
        return {'insight_verification': result, 'stage': PipelineStage.VERIFICATION, 'trace': self._trace(state, 'verifier'), 'step_logs': self._append_log(state, stage='verifier', title='Verification', summary=result.insight_verification_summary, inputs=['insight candidates'], outputs=result.verified_insights[:3], details=self._stage_details(before) | result.model_dump())}

    @traceable(name='virage.insights_node')
    def insights_node(self, state: PipelineState) -> dict:
        result = self.insights.invoke(state['insight_verification'])
        return {'insights': result, 'stage': PipelineStage.INSIGHTS, 'trace': self._trace(state, 'insights'), 'step_logs': self._append_log(state, stage='insights', title='Final insights', summary=f'{len(result.final_insights)} insights', inputs=['verified insights'], outputs=result.final_insights[:3], details=result.model_dump())}

    @traceable(name='virage.vision_score_node')
    def vision_score_node(self, state: PipelineState) -> dict:
        if not self.runtime.settings.enable_vision_score:
            return {}
        before = len(self.runtime.model_call_logs)
        plot_image = PlotImageArtifact(**state['plot_image'])
        result = self.vision_score.invoke(plot_image, runtime=self.runtime, query_understanding=state.get('query_understanding'))
        return {'visual_quality_metric': result, 'stage': PipelineStage.EVALUATION, 'trace': self._trace(state, 'vision_score'), 'step_logs': self._append_log(state, stage='vision_score', title='Visual metric', summary=f'score={result.score:.3f}', inputs=[state['plot_image']['image_path']], outputs=result.details[:3], details=self._stage_details(before) | result.model_dump())}

    @traceable(name='virage.evaluation_summary_node')
    def evaluation_summary_node(self, state: PipelineState) -> dict:
        if not self.runtime.settings.enable_evaluation_summary:
            return {}
        result = self.evaluation_summary.invoke(state.get('structural_spec_metric'), state.get('visual_quality_metric'), state['empty_chart_check'], state['insight_verification'])
        return {'evaluation_summary': result, 'stage': PipelineStage.EVALUATION, 'trace': self._trace(state, 'evaluation_summary'), 'step_logs': self._append_log(state, stage='evaluation_summary', title='Evaluation summary', summary='Aggregated metrics and verification results', inputs=['metrics', 'verification'], outputs=[str(result.benchmark_report)], details=result.model_dump())}
