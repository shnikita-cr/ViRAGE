from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.infrastructure.runtime import RuntimeContext
from src.services.artifact_store import ArtifactStoreService
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
        self.artifact_store = ArtifactStoreService()

    @staticmethod
    def _trace(state: PipelineState, label: str) -> list[str]:
        return [*state.get("trace", []), label]

    def query_understanding_node(self, state: PipelineState) -> dict:
        result = self.query_understanding.invoke(state["query"], state["user_context"], runtime=self.runtime)
        return {
            "query_understanding": result,
            "query_intent_bundle": result.to_intent_bundle(),
            "stage": PipelineStage.QUERY_UNDERSTANDING,
            "trace": self._trace(state, "query_understanding"),
        }

    def data_profiler_node(self, state: PipelineState) -> dict:
        result = self.data_profiler.invoke(state["data_path"], runtime=self.runtime)
        return {"data_profile": result, "stage": PipelineStage.DATA_PROFILING, "trace": self._trace(state, "data_profiler")}

    def request_analyzer_node(self, state: PipelineState) -> dict:
        result = self.request_analyzer.invoke(
            query=state["query"],
            query_understanding=state["query_understanding"],
            data_profile=state["data_profile"],
            runtime=self.runtime,
        )
        return {"request_analysis": result, "stage": PipelineStage.REQUEST_ANALYSIS, "trace": self._trace(state, "request_analyzer")}

    def data_preparation_node(self, state: PipelineState) -> dict:
        result = self.data_preparation.invoke(
            state["data_path"],
            state["data_profile"],
            state["request_analysis"],
            state["run_id"],
            runtime=self.runtime,
        )
        return {"data_preparation": result, "stage": PipelineStage.DATA_PREPARATION, "trace": self._trace(state, "data_preparation")}

    def visrag_node(self, state: PipelineState) -> dict:
        result = self.visrag.invoke(state["query_understanding"], state["request_analysis"], state["data_profile"], runtime=self.runtime)
        return {
            "visrag": result,
            "candidate_spec_set": result.candidate_spec_set,
            "stage": PipelineStage.VISRAG,
            "trace": self._trace(state, "visrag"),
        }

    def planning_node(self, state: PipelineState) -> dict:
        result = self.planning.invoke(
            state["query_understanding"],
            state["request_analysis"],
            state["data_profile"],
            state["visrag"],
            runtime=self.runtime,
        )
        return {
            "planning": result,
            "execution_policy": result.execution_policy,
            "validation_policy": result.validation_policy,
            "analysis_rubric": result.analysis_rubric,
            "stage": PipelineStage.PLANNING,
            "trace": self._trace(state, "planning"),
        }

    def chart_generator_node(self, state: PipelineState) -> dict:
        result = self.chart_generator.invoke(
            state["data_preparation"],
            state["candidate_spec_set"],
            state["execution_policy"],
            state["validation_policy"],
            runtime=self.runtime,
        )
        return {"vega_spec": result, "stage": PipelineStage.CHART_GENERATION, "trace": self._trace(state, "chart_generator")}

    def spec_validator_node(self, state: PipelineState) -> dict:
        max_retries = state.get("execution_policy").max_retries if state.get("execution_policy") else 0
        current_spec = state["vega_spec"]
        attempts = 0
        latest = self.spec_validator.invoke(current_spec)
        while not latest.is_valid and attempts < max_retries:
            attempts += 1
            current_spec = self.chart_generator.repair(
                state["data_preparation"],
                current_spec,
                latest.validation_errors,
                latest.repair_hints,
                runtime=self.runtime,
            )
            latest = self.spec_validator.invoke(current_spec)
        payload = {
            "spec_validation": latest,
            "stage": PipelineStage.SPEC_VALIDATION,
            "trace": self._trace(state, "spec_validator"),
        }
        if current_spec is not state["vega_spec"]:
            payload["vega_spec"] = current_spec
        return payload

    def vegalite_plot_drawing_node(self, state: PipelineState) -> dict:
        result = self.vegalite_plot_drawing.invoke(state["spec_validation"], state["run_id"], runtime=self.runtime)
        return {
            "plot_rendering": result,
            "plot_image": result.plot_image.model_dump(),
            "stage": PipelineStage.PLOT_RENDERING,
            "trace": self._trace(state, "vegalite_plot_drawing"),
        }

    def scenegraph_check_node(self, state: PipelineState) -> dict:
        result = self.scenegraph_check.invoke(state["plot_rendering"])
        return {"scenegraph_check": result, "stage": PipelineStage.SCENEGRAPH_CHECK, "trace": self._trace(state, "scenegraph_check")}

    def empty_chart_check_node(self, state: PipelineState) -> dict:
        result = self.empty_chart_check.invoke(state["scenegraph_check"])
        current_spec = state["vega_spec"]
        current_validation = state["spec_validation"]
        current_rendering = state["plot_rendering"]
        current_scenegraph = state["scenegraph_check"]
        candidate_set = state["candidate_spec_set"]
        policy = state.get("execution_policy")
        max_retries = policy.max_retries if policy else 0
        attempts = 0

        while result.empty_chart_signal and attempts < max_retries and candidate_set and policy and policy.fallback_enabled:
            attempts += 1
            current_index = 0
            selected = candidate_set.selected_candidate_spec
            if selected is not None:
                for idx, candidate in enumerate(candidate_set.candidate_specs):
                    if candidate.spec_id == selected.spec_id:
                        current_index = idx
                        break
            next_index = current_index + 1
            if next_index >= len(candidate_set.candidate_specs):
                break
            candidate_set = candidate_set.model_copy(deep=True)
            candidate_set.selected_candidate_spec = candidate_set.candidate_specs[next_index]
            if candidate_set.selected_candidate_spec.visualization_plan is not None:
                candidate_set.visualization_plan = candidate_set.selected_candidate_spec.visualization_plan
            current_spec = self.chart_generator.build_from_candidate(
                state["data_preparation"],
                candidate_set,
                next_index,
                state["execution_policy"],
                state["validation_policy"],
                runtime=self.runtime,
            )
            current_validation = self.spec_validator.invoke(current_spec)
            if not current_validation.is_valid:
                continue
            current_rendering = self.vegalite_plot_drawing.invoke(current_validation, state["run_id"], runtime=self.runtime)
            current_scenegraph = self.scenegraph_check.invoke(current_rendering)
            result = self.empty_chart_check.invoke(current_scenegraph)

        payload = {
            "empty_chart_check": result,
            "stage": PipelineStage.EMPTY_CHART_CHECK,
            "trace": self._trace(state, "empty_chart_check"),
        }
        if current_spec is not state["vega_spec"]:
            payload.update({
                "candidate_spec_set": candidate_set,
                "vega_spec": current_spec,
                "spec_validation": current_validation,
                "plot_rendering": current_rendering,
                "plot_image": current_rendering.plot_image.model_dump(),
                "scenegraph_check": current_scenegraph,
            })
        return payload

    def vlm_analysis_node(self, state: PipelineState) -> dict:
        from src.domain.models import PlotImageArtifact
        plot_image = PlotImageArtifact(**state["plot_image"])
        result = self.vlm_analysis.invoke(plot_image, state["analysis_rubric"], runtime=self.runtime)
        return {"vlm_analysis": result, "stage": PipelineStage.VLM_ANALYSIS, "trace": self._trace(state, "vlm_analysis")}

    def fact_extractor_node(self, state: PipelineState) -> dict:
        result = self.fact_extractor.invoke(state["vlm_analysis"], runtime=self.runtime)
        return {"visual_facts": result, "stage": PipelineStage.FACT_EXTRACTION, "trace": self._trace(state, "fact_extractor")}

    def reasoner_node(self, state: PipelineState) -> dict:
        result = self.reasoner.invoke(state["visual_facts"], state["analysis_rubric"], runtime=self.runtime)
        return {"insight_reasoning": result, "stage": PipelineStage.REASONING, "trace": self._trace(state, "reasoner")}

    def verifier_node(self, state: PipelineState) -> dict:
        result = self.verifier.invoke(state["insight_reasoning"], runtime=self.runtime)
        return {"insight_verification": result, "stage": PipelineStage.VERIFICATION, "trace": self._trace(state, "verifier")}

    def insights_node(self, state: PipelineState) -> dict:
        result = self.insights.invoke(state["insight_verification"])
        return {"insights": result, "stage": PipelineStage.INSIGHTS, "trace": self._trace(state, "insights")}

    def spec_score_node(self, state: PipelineState) -> dict:
        result = self.spec_score.invoke(state["spec_validation"])
        return {"structural_spec_metric": result, "stage": PipelineStage.EVALUATION, "trace": self._trace(state, "spec_score")}

    def vision_score_node(self, state: PipelineState) -> dict:
        from src.domain.models import PlotImageArtifact
        plot_image = PlotImageArtifact(**state["plot_image"])
        result = self.vision_score.invoke(plot_image, runtime=self.runtime)
        return {"visual_quality_metric": result, "stage": PipelineStage.EVALUATION, "trace": self._trace(state, "vision_score")}

    def evaluation_summary_node(self, state: PipelineState) -> dict:
        result = self.evaluation_summary.invoke(
            state["structural_spec_metric"],
            state["visual_quality_metric"],
            state["empty_chart_check"],
            state["insight_verification"],
        )
        return {"evaluation_summary": result, "stage": PipelineStage.EVALUATION, "trace": self._trace(state, "evaluation_summary")}
