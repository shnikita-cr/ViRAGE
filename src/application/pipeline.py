from src.application.contracts import PipelineRequest, PipelineResult
from src.application.settings import ViRAGESettings
from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.graph.builder import build_pipeline_graph
from src.infrastructure.runtime import RuntimeContext


class ViRAGEPipeline:
    def __init__(self, settings: ViRAGESettings | None = None, llm: object | None = None) -> None:
        self.settings = settings or ViRAGESettings()
        self.runtime = RuntimeContext(settings=self.settings, llm=llm)
        self.graph = build_pipeline_graph(self.runtime)

    def invoke(self, request: PipelineRequest) -> PipelineResult:
        initial_state: PipelineState = {
            "run_id": request.run_id,
            "query": request.query,
            "data_path": request.data_path,
            "user_context": request.user_context,
            "stage": PipelineStage.INITIALIZED,
            "trace": [],
            "errors": [],
        }
        final_state: PipelineState = self.graph.invoke(initial_state)
        final_state["stage"] = PipelineStage.COMPLETED
        return PipelineResult(
            run_id=final_state["run_id"],
            query=final_state["query"],
            data_path=final_state["data_path"],
            case_type=final_state["case_type"],
            query_understanding=final_state["query_understanding"],
            planning=final_state["planning"],
            data_profile=final_state["data_profile"],
            data_preparation=final_state["data_preparation"],
            visrag=final_state["visrag"],
            codegen=final_state["codegen"],
            execution=final_state["execution"],
            artifact_bundle=final_state["artifact_bundle"],
            chart_read=final_state["chart_read"],
            facts=final_state["facts"],
            reasoning=final_state["reasoning"],
            verification=final_state["verification"],
        )
