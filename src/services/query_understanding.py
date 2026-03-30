from typing import Any

from pydantic import BaseModel
from src.domain.enums import ChartCaseType
from src.domain.models import QueryUnderstandingResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured, is_langchain_available
from src.services.base import BaseService


class _QueryUnderstandingSchema(BaseModel):
    intent: str
    requested_operations: list[str]
    candidate_charts: list[str]
    constraints: list[str]
    case_type: ChartCaseType
    confidence: float


class QueryUnderstandingService(BaseService):
    def invoke(self, query: str, user_context: dict[str, Any], runtime: RuntimeContext) -> QueryUnderstandingResult:
        if runtime.llm is not None and is_langchain_available():
            prompt = (
                "Extract chart-analysis intent from the request and return structured output.\n"
                f"Request: {query}\nContext: {user_context}"
            )
            try:
                parsed = invoke_structured(runtime.llm, prompt, _QueryUnderstandingSchema)
                return QueryUnderstandingResult(**parsed.model_dump())
            except Exception:
                pass

        q = query.lower()
        charts: list[str] = []
        ops: list[str] = []
        constraints: list[str] = []
        if any(t in q for t in ["trend", "time", "timeline", "over time"]):
            charts.append("line")
            ops.append("trend analysis")
        if any(t in q for t in ["distribution", "spread", "outlier", "hist"]):
            charts.extend(["histogram", "boxplot"])
            ops.append("distribution analysis")
        if any(t in q for t in ["compare", "comparison", "vs", "versus"]):
            charts.append("bar")
            ops.append("comparison")
        if any(t in q for t in ["correlation", "relationship", "scatter"]):
            charts.append("scatter")
            ops.append("relationship analysis")
        if "3d" in q:
            constraints.append("avoid 3d unless strictly required")
        if not charts:
            charts = ["bar", "line", "scatter"]
        if not ops:
            ops = ["exploratory analysis"]
        case_type = ChartCaseType.NON_CANONICAL if any(
            t in q for t in ["diagram", "flowchart", "infographic", "network"]) else ChartCaseType.CANONICAL
        return QueryUnderstandingResult(
            intent=query.strip(),
            requested_operations=list(dict.fromkeys(ops)),
            candidate_charts=list(dict.fromkeys(charts)),
            constraints=constraints,
            case_type=case_type,
            confidence=0.65 if case_type is ChartCaseType.CANONICAL else 0.45,
        )
