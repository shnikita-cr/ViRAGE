from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.application.settings import ViRAGESettings
from src.domain.models import (
    DataColumnProfile,
    DataPreparationResult,
    DataProfile,
    QueryRequestAnalysisResult,
    VisRAGGenerationGuidance,
    VisRAGResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.chart_generator import ChartGeneratorService


@dataclass
class FakeLLMResult:
    content: str
    usage_metadata: dict[str, int]


class FakeSpecLLM:
    model = "fake-spec-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[object] = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return FakeLLMResult(
            content=self.response,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def test_chart_generator_uses_vegachat_codegen_backend_and_safe_fields(tmp_path: Path) -> None:
    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
            spec_generation_backend="vegachat_codegen",
            spec_generation_response_parse_retries=0,
        ),
        spec_llm=FakeSpecLLM(
            "<explain>Compare mean metric by region.</explain>"
            "<json>{\"$schema\":\"https://vega.github.io/schema/vega-lite/v5.json\","
            "\"mark\":\"bar\","
            "\"encoding\":{" 
            "\"x\":{\"field\":\"Region_Name\",\"type\":\"nominal\"},"
            "\"y\":{\"field\":\"Metric_Value\",\"type\":\"quantitative\",\"aggregate\":\"mean\"}"
            "}}</json>"
        ),
    )
    runtime.current_run_id = "run"
    runtime.ensure_run_dir()

    prepared = DataPreparationResult(
        output_path="prepared.csv",
        row_count=2,
        col_count=2,
        column_name_map={"Region.Name": "Region_Name", "Metric Value (%)": "Metric_Value"},
        reverse_column_name_map={"Region_Name": "Region.Name", "Metric_Value": "Metric Value (%)"},
        original_columns=["Region.Name", "Metric Value (%)"],
        safe_columns=["Region_Name", "Metric_Value"],
        renamed_column_count=2,
    )
    profile = DataProfile(
        row_count=2,
        col_count=2,
        columns=[
            DataColumnProfile(name="Region.Name", safe_name="Region_Name",
                              dtype="categorical", role="dimension", missing_ratio=0.0, unique_count=2),
            DataColumnProfile(name="Metric Value (%)", safe_name="Metric_Value",
                              dtype="numeric", role="measure", missing_ratio=0.0, unique_count=2),
        ],
    )
    visrag = VisRAGResult(
        generation_guidance=VisRAGGenerationGuidance(
            prompt_text="VisRAG rule guidance:\n- Use a bar chart for category comparison."
        )
    )

    artifact = ChartGeneratorService().invoke(
        prepared=prepared,
        runtime=runtime,
        query="compare metric by region",
        data_profile=profile,
        query_request_analysis=QueryRequestAnalysisResult(
            normalized_query="compare metric by region",
            analysis_task="comparison",
            recommended_chart_family="bar",
            selected_fields=["Region.Name", "Metric Value (%)"],
        ),
        visrag=visrag,
    )

    assert artifact.generation_backend == "vegachat_codegen"
    assert artifact.generation_explanation == "Compare mean metric by region."
    assert artifact.spec_json["data"] == {"url": "prepared.csv"}
    assert "data" not in artifact.spec_without_runtime_data
    assert artifact.spec_without_runtime_data["encoding"]["x"]["field"] == "Region_Name"
    assert artifact.spec_json["encoding"]["x"]["field"] == "Region_Name"
    assert artifact.spec_json["encoding"]["y"]["field"] == "Metric_Value"
    assert "VisRAG rule guidance" in str(runtime.spec_llm.prompts[0])
    assert (tmp_path / "artifacts" / "run" / "model_calls.csv").exists()
    assert list((tmp_path / "artifacts" / "run" / "model_calls").glob("*_chart_generator_spec-01.json"))
