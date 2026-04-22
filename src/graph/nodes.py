from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.infrastructure.runtime import RuntimeContext
from src.services import (
    ArtifactStoreService,
    ChartReaderService,
    CodeRunService,
    CodegenService,
    DataPreparationService,
    DataProfilerService,
    FactExtractorService,
    PlanningService,
    QueryUnderstandingService,
    ReasonerService,
    RequestAnalyzerService,
    VerifierService,
    VisRAGService,
)


class PipelineNodes:
    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.query_understanding = QueryUnderstandingService()
        self.data_profiler = DataProfilerService()
        self.request_analyzer = RequestAnalyzerService()
        self.data_preparation = DataPreparationService()
        self.visrag = VisRAGService()
        self.planning = PlanningService()
        self.codegen = CodegenService()
        self.coderun = CodeRunService()
        self.artifact_store = ArtifactStoreService()
        self.chart_reader = ChartReaderService()
        self.fact_extractor = FactExtractorService()
        self.reasoner = ReasonerService()
        self.verifier = VerifierService()

    def _trace(self, state: PipelineState, step: str) -> list[str]:
        return [*state.get("trace", []), step]

    def query_understanding_node(self, state: PipelineState) -> dict:
        result = self.query_understanding.invoke(state["query"], state.get("user_context", {}), runtime=self.runtime)
        return {
            "query_understanding": result,
            "query_intent_bundle": result.to_intent_bundle(),
            "case_type": None,
            "stage": PipelineStage.QUERY_UNDERSTANDING,
            "trace": self._trace(state, "query_understanding"),
        }

    def data_profiler_node(self, state: PipelineState) -> dict:
        result = self.data_profiler.invoke(state["data_path"], runtime=self.runtime)
        return {
            "data_profile": result,
            "stage": PipelineStage.DATA_PROFILING,
            "trace": self._trace(state, "data_profiler"),
        }

    def request_analyzer_node(self, state: PipelineState) -> dict:
        result = self.request_analyzer.invoke(
            query=state["query"],
            query_understanding=state["query_understanding"],
            data_profile=state["data_profile"],
            runtime=self.runtime,
        )
        return {
            "request_analysis": result,
            "stage": PipelineStage.REQUEST_ANALYSIS,
            "trace": self._trace(state, "request_analyzer"),
        }

    def data_preparation_node(self, state: PipelineState) -> dict:
        result = self.data_preparation.invoke(
            state["data_path"],
            state["data_profile"],
            state["request_analysis"],
            state["run_id"],
            runtime=self.runtime,
        )
        return {
            "data_preparation": result,
            "stage": PipelineStage.DATA_PREPARATION,
            "trace": self._trace(state, "data_preparation"),
        }

    def visrag_node(self, state: PipelineState) -> dict:
        result = self.visrag.invoke(
            state["query_understanding"],
            state["request_analysis"],
            state["data_profile"],
            runtime=self.runtime,
        )
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

    def codegen_node(self, state: PipelineState) -> dict:
        result = self.codegen.invoke(
            state["query_understanding"],
            state["planning"],
            state["data_profile"],
            state["data_preparation"],
            state["visrag"],
            state["run_id"],
            runtime=self.runtime,
        )
        return {"codegen": result, "stage": PipelineStage.CODEGEN, "trace": self._trace(state, "codegen")}

    def coderun_node(self, state: PipelineState) -> dict:
        result = self.coderun.invoke(state["codegen"], state["run_id"], runtime=self.runtime)
        return {"execution": result, "stage": PipelineStage.CODERUN, "trace": self._trace(state, "coderun")}

    def artifact_store_node(self, state: PipelineState) -> dict:
        result = self.artifact_store.invoke(state["data_preparation"], state["execution"], state["run_id"], runtime=self.runtime)
        return {
            "artifact_bundle": result,
            "stage": PipelineStage.ARTIFACT_STORE,
            "trace": self._trace(state, "artifact_store"),
        }

    def chart_reader_node(self, state: PipelineState) -> dict:
        result = self.chart_reader.invoke(state["execution"], runtime=self.runtime)
        return {
            "chart_read": result,
            "stage": PipelineStage.CHART_READING,
            "trace": self._trace(state, "chart_reader"),
        }

    def fact_extractor_node(self, state: PipelineState) -> dict:
        result = self.fact_extractor.invoke(state["execution"], state["chart_read"], runtime=self.runtime)
        return {
            "facts": result,
            "stage": PipelineStage.FACT_EXTRACTION,
            "trace": self._trace(state, "fact_extractor"),
        }

    def reasoner_node(self, state: PipelineState) -> dict:
        result = self.reasoner.invoke(state["facts"], runtime=self.runtime)
        return {"reasoning": result, "stage": PipelineStage.REASONING, "trace": self._trace(state, "reasoner")}

    def verifier_node(self, state: PipelineState) -> dict:
        result = self.verifier.invoke(state["reasoning"], runtime=self.runtime)
        return {
            "verification": result,
            "stage": PipelineStage.VERIFICATION,
            "trace": self._trace(state, "verifier"),
        }
