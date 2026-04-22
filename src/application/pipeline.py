from src.application.contracts import PipelineRequest, PipelineResult
from src.application.settings import ViRAGESettings
from src.application.state import PipelineState
from src.domain.enums import PipelineStage
from src.graph.builder import build_pipeline_graph
from src.infrastructure.runtime import RuntimeContext


class ViRAGEPipeline:
    def __init__(
            self,
            settings: ViRAGESettings | None = None,
            reasoning_llm: object | None = None,
            codegen_llm: object | None = None,
            spec_llm: object | None = None,
            vlm: object | None = None,
            vision_judge_llm: object | None = None,
    ) -> None:
        self.settings = settings or ViRAGESettings()
        self.runtime = RuntimeContext(
            settings=self.settings,
            reasoning_llm=reasoning_llm,
            codegen_llm=codegen_llm,
            spec_llm=spec_llm,
            vlm=vlm,
            vision_judge_llm=vision_judge_llm,
        )
        self.graph = build_pipeline_graph(self.runtime)

    def invoke(self, request: PipelineRequest) -> PipelineResult:
        initial_state: PipelineState = {
            "run_id": request.run_id,
            "query": request.query,
            "data_path": request.data_path,
            "case_type": None,
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
            case_type=final_state.get("case_type"),
            query_understanding=final_state.get("query_understanding"),
            planning=final_state.get("planning"),
            data_profile=final_state.get("data_profile"),
            data_preparation=final_state.get("data_preparation"),
            visrag=final_state.get("visrag"),
            query_intent_bundle=final_state.get("query_intent_bundle"),
            request_analysis=final_state.get("request_analysis"),
            execution_policy=final_state.get("execution_policy"),
            validation_policy=final_state.get("validation_policy"),
            analysis_rubric=final_state.get("analysis_rubric"),
            candidate_spec_set=final_state.get("candidate_spec_set"),
            vega_spec=final_state.get("vega_spec"),
            spec_validation=final_state.get("spec_validation"),
            plot_rendering=final_state.get("plot_rendering"),
            scenegraph_check=final_state.get("scenegraph_check"),
            empty_chart_check=final_state.get("empty_chart_check"),
            plot_image=final_state.get("plot_image"),
            vlm_analysis=final_state.get("vlm_analysis"),
            visual_facts=final_state.get("visual_facts"),
            insight_reasoning=final_state.get("insight_reasoning"),
            insight_verification=final_state.get("insight_verification"),
            insights=final_state.get("insights"),
            structural_spec_metric=final_state.get("structural_spec_metric"),
            visual_quality_metric=final_state.get("visual_quality_metric"),
            evaluation_summary=final_state.get("evaluation_summary"),
        )
