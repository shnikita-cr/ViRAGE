from src.domain.models import PlanningResult, PlanningStep, QueryUnderstandingResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class PlanningService(BaseService):
    def invoke(self, query_understanding: QueryUnderstandingResult, runtime: RuntimeContext) -> PlanningResult:
        steps = [
            PlanningStep(name="profile_data", description="Inspect dataset structure and quality."),
            PlanningStep(name="prepare_data", description="Apply minimal cleaning and normalization."),
            PlanningStep(name="retrieve_visual_knowledge", description="Retrieve charting guidance for the task."),
            PlanningStep(name="generate_and_run", description="Generate and execute plotting code."),
            PlanningStep(name="analyze_chart", description="Read chart structure and derive facts."),
            PlanningStep(name="verify", description="Validate conclusions against artifacts and metrics."),
        ]
        criteria = [
            "At least one chart is generated successfully.",
            "Artifacts are saved and traceable.",
            "Final statements include evidence references.",
        ]
        return PlanningResult(mode=query_understanding.case_type, steps=steps, success_criteria=criteria)
