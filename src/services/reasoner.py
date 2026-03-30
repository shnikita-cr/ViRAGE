from src.domain.models import FactExtractionResult, ReasoningResult, ReasoningStatement
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class ReasonerService(BaseService):
    def invoke(self, facts: FactExtractionResult, runtime: RuntimeContext) -> ReasoningResult:
        mapping = {f.name: f for f in facts.facts}
        statements = []
        if "chart_type" in mapping:
            statements.append(ReasoningStatement(text=f"The selected chart type is {mapping['chart_type'].value}.",
                                                 evidence=mapping["chart_type"].evidence))
        if {"y_min", "y_max", "y_mean"}.issubset(mapping):
            statements.append(ReasoningStatement(
                text=f"The primary numeric measure ranges from {mapping['y_min'].value} to {mapping['y_max'].value} with mean {mapping['y_mean'].value}.",
                evidence=mapping["y_mean"].evidence))
        summary = statements[0].text if statements else "No reasoning statements were derived."
        return ReasoningResult(summary=summary, statements=statements)
