from __future__ import annotations

from src.services.chart_reader import ChartReaderService


def test_chart_reader_service_reads_basic_chart_structure(code_run_result, runtime) -> None:
    service = ChartReaderService()

    result = service.invoke(execution=code_run_result, runtime=runtime)

    assert result.chart_type == "line"
    assert result.axes == {"x": "date", "y": "sales"}
    assert result.source_artifact is not None
