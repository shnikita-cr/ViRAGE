from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.infrastructure.runtime import RuntimeContext
from src.services import (
    ArtifactStoreService,
    CanonicalPlanningService,
    ChartReaderService,
    CodeRunService,
    CodegenService,
    DataPreparationService,
    DataProfilerService,
    FactExtractorService,
    NonCanonicalPlanningService,
    QueryUnderstandingService,
    ReasonerService,
    VerifierService,
    VisRAGService,
)


class PipelineNodes:
    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.query_understanding = QueryUnderstandingService()
        self.planning_canonical = CanonicalPlanningService()
        self.planning_non_canonical = NonCanonicalPlanningService()
        self.data_profiler = DataProfilerService()
        self.data_preparation = DataPreparationService()
        self.visrag = VisRAGService()
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
            "case_type": result.case_type,
            "stage": PipelineStage.QUERY_UNDERSTANDING,
            "trace": self._trace(state, "query_understanding"),
        }

    def planning_canonical_node(self, state: PipelineState) -> dict:
        result = self.planning_canonical.invoke(state["query_understanding"], runtime=self.runtime)
        return {
            "planning": result,
            "stage": PipelineStage.PLANNING,
            "trace": self._trace(state, "planning_canonical"),
        }

    def planning_non_canonical_node(self, state: PipelineState) -> dict:
        result = self.planning_non_canonical.invoke(state["query_understanding"], runtime=self.runtime)
        return {
            "planning": result,
            "stage": PipelineStage.PLANNING,
            "trace": self._trace(state, "planning_non_canonical"),
        }

    def data_profiler_node(self, state: PipelineState) -> dict:
        result = self.data_profiler.invoke(state["data_path"], runtime=self.runtime)
        return {
            "data_profile": result,
            "stage": PipelineStage.DATA_PROFILING,
            "trace": self._trace(state, "data_profiler"),
        }

    def data_preparation_node(self, state: PipelineState) -> dict:
        result = self.data_preparation.invoke(state["data_path"], state["data_profile"], state["run_id"],
                                              runtime=self.runtime)
        return {
            "data_preparation": result,
            "stage": PipelineStage.DATA_PREPARATION,
            "trace": self._trace(state, "data_preparation"),
        }

    def visrag_node(self, state: PipelineState) -> dict:
        result = self.visrag.invoke(state["query_understanding"], state["data_profile"], runtime=self.runtime)
        return {
            "visrag": result,
            "stage": PipelineStage.VISRAG,
            "trace": self._trace(state, "visrag"),
        }

    def codegen_node(self, state: PipelineState) -> dict:
        result = self.codegen.invoke(
            state["query_understanding"],
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
        result = self.artifact_store.invoke(state["data_preparation"], state["execution"], state["run_id"],
                                            runtime=self.runtime)
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
