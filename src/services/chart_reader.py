from src.domain.models import ChartElement, ChartReadResult, CodeRunResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class ChartReaderService(BaseService):
    def invoke(self, execution: CodeRunResult, runtime: RuntimeContext) -> ChartReadResult:
        metadata = execution.chart_metadata
        x_column = metadata.get("x_column")
        y_column = metadata.get("y_column")
        plot_path = next((a.path for a in execution.artifacts if a.artifact_type.value == "plot"), None)
        return ChartReadResult(
            chart_type=str(metadata.get("chart_type", "unknown")),
            title=str(metadata.get("title")) if metadata.get("title") is not None else None,
            axes={"x": str(x_column or "index"), "y": str(y_column or "value")},
            series=[str(y_column)] if y_column else [],
            elements=[ChartElement(kind="axis", label="x", value=x_column),
                      ChartElement(kind="axis", label="y", value=y_column)],
            source_artifact=plot_path,
        )
