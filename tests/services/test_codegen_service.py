from __future__ import annotations

from src.services.codegen import CodegenService


def test_codegen_service_generates_python_plot_code(
    runtime,
    canonical_query_understanding,
    sample_data_profile,
    prepared_result,
    visrag_result,
) -> None:
    service = CodegenService()

    result = service.invoke(
        query_understanding=canonical_query_understanding,
        data_profile=sample_data_profile,
        prepared=prepared_result,
        visrag=visrag_result,
        run_id="codegen",
        runtime=runtime,
    )

    assert result.language == "python"
    assert result.chart_type == "line"
    assert "pd.read_csv" in result.code
    assert "sales" in result.code
    assert "date" in result.code
