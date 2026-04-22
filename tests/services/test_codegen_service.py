from __future__ import annotations

from pathlib import Path

import pytest

from src.application.settings import ViRAGESettings
from src.domain.models import PlanningResult, PlanningStep, QueryUnderstandingResult
from src.infrastructure.runtime import RuntimeContext
from src.services.codegen import CodegenService
from tests._fakes import FakeCodegenLLM, LINE_PLOT_LOGIC


def test_codegen_requires_codegen_llm(
        runtime: RuntimeContext,
        canonical_query_understanding: QueryUnderstandingResult,
        sample_data_profile,
        prepared_result,
        visrag_result,
) -> None:
    planning = PlanningResult(mode=None, steps=[PlanningStep(name="build_chart", description="Build a simple chart.")])
    with pytest.raises(RuntimeError, match="codegen_llm"):
        CodegenService().invoke(
            canonical_query_understanding,
            planning,
            sample_data_profile,
            prepared_result,
            visrag_result,
            run_id="run-no-llm",
            runtime=runtime,
        )


def test_codegen_generates_executable_wrapper_and_trace_artifacts(
        tmp_path: Path,
        canonical_query_understanding: QueryUnderstandingResult,
        sample_data_profile,
        prepared_result,
        visrag_result,
) -> None:
    fake_llm = FakeCodegenLLM(f"```python\n{LINE_PLOT_LOGIC}\n```")
    runtime = RuntimeContext(
        settings=ViRAGESettings(artifact_root=tmp_path / "artifacts", codegen_store_trace_artifacts=True),
        codegen_llm=fake_llm,
    )
    planning = PlanningResult(mode=None, steps=[PlanningStep(name="build_chart", description="Build a simple chart.")])

    result = CodegenService().invoke(
        canonical_query_understanding,
        planning,
        sample_data_profile,
        prepared_result,
        visrag_result,
        run_id="run-codegen",
        runtime=runtime,
    )

    assert result.chart_type == "line"
    assert "def main(" in result.code
    assert "plot_built = False" in result.code
    assert "ax.plot(" in result.code
    assert result.prompt_path is not None and Path(result.prompt_path).exists()
    assert result.raw_response_path is not None and Path(result.raw_response_path).exists()
    assert result.generated_logic_path is not None and Path(result.generated_logic_path).exists()
    assert fake_llm.last_prompt is not None
    assert "Visualization plan JSON" in fake_llm.last_prompt
