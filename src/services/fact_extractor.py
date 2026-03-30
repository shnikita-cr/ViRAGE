from src.domain.models import ChartReadResult, CodeRunResult, Fact, FactExtractionResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class FactExtractorService(BaseService):
    def invoke(self, execution: CodeRunResult, chart_read: ChartReadResult,
               runtime: RuntimeContext) -> FactExtractionResult:
        metrics = {m.name: m.value for m in execution.metrics}
        evidence = [a.path for a in execution.artifacts]
        facts = [Fact(name="chart_type", value=chart_read.chart_type, evidence=evidence)]
        for key in ["row_count", "column_count", "x_column", "y_column", "y_min", "y_max", "y_mean", "group_count"]:
            if key in metrics:
                facts.append(Fact(name=key, value=str(metrics[key]), evidence=evidence))
        if chart_read.chart_type == "line" and "y_min" in metrics and "y_max" in metrics:
            facts.append(Fact(name="trend_hint", value="numeric series rendered as line", evidence=evidence))
        return FactExtractionResult(facts=facts)
