from src.application.state import PipelineState
from src.domain.enums import ChartCaseType


def route_case(state: PipelineState) -> str:
    return "planning_non_canonical" \
        if state["query_understanding"].case_type is ChartCaseType.NON_CANONICAL \
        else "planning_canonical"
