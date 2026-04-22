from __future__ import annotations

from pathlib import Path

from src.application.settings import ViRAGESettings
from src.domain.models import PlanningResult, PlanningStep, QueryUnderstandingResult
from src.infrastructure.runtime import RuntimeContext
from src.services.codegen import CodegenService
from src.services.coderun import CodeRunService
from tests._fakes import FakeCodegenLLM, LINE_PLOT_LOGIC


def test_coderun_executes_llm_generated_wrapper(
        tmp_path: Path,
        canonical_query_understanding: QueryUnderstandingResult,
        sample_data_profile,
        prepared_result,
        visrag_result,
) -> None:
    runtime = RuntimeContext(
        settings=ViRAGESettings(artifact_root=tmp_path / "artifacts", codegen_store_trace_artifacts=True),
        codegen_llm=FakeCodegenLLM(LINE_PLOT_LOGIC),
    )
    planning = PlanningResult(mode=None, steps=[PlanningStep(name="build_chart", description="Build a simple chart.")])
    codegen_result = CodegenService().invoke(
        canonical_query_understanding,
        planning,
        sample_data_profile,
        prepared_result,
        visrag_result,
        run_id="run-coderun",
        runtime=runtime,
    )

    execution = CodeRunService().invoke(codegen_result, run_id="run-coderun", runtime=runtime)

    assert execution.success is True
    assert any(artifact.artifact_type.value == "plot" for artifact in execution.artifacts)
    assert execution.chart_metadata.get("chart_type") == "line"
    metric_names = {metric.name for metric in execution.metrics}
    assert {"row_count", "column_count", "x_column", "y_column"}.issubset(metric_names)
