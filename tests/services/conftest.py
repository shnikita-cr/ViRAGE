from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.settings import ViRAGESettings
from src.domain.enums import ArtifactType, ChartCaseType
from src.domain.models import (
    ArtifactRef,
    ChartReadResult,
    CodeRunResult,
    DataPreparationResult,
    DataProfile,
    ExecutionMetric,
    Fact,
    FactExtractionResult,
    QueryUnderstandingResult,
    ReasoningResult,
    ReasoningStatement,
    VisualizationAxisInstruction,
    VisualizationFieldBinding,
    VisualizationPlan,
    VisRAGRecommendation,
    VisRAGResult,
)
from src.infrastructure.runtime import RuntimeContext


@pytest.fixture
def runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))


@pytest.fixture
def canonical_query_understanding() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        intent="Show sales trend over time",
        requested_operations=["trend analysis"],
        candidate_charts=["line", "bar"],
        constraints=[],
        case_type=ChartCaseType.CANONICAL,
        confidence=0.9,
    )


@pytest.fixture
def non_canonical_query_understanding() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        intent="Build a network diagram of flows",
        requested_operations=["relationship analysis"],
        candidate_charts=["scatter", "bar"],
        constraints=["prefer concise visuals"],
        case_type=ChartCaseType.NON_CANONICAL,
        confidence=0.6,
    )


@pytest.fixture
def sample_data_profile() -> DataProfile:
    return DataProfile(
        row_count=4,
        col_count=3,
        columns=[],
        likely_numeric_columns=["sales", "profit"],
        likely_categorical_columns=["region"],
        likely_time_columns=["date"],
        quality_notes=[],
    )


@pytest.fixture
def prepared_result(tmp_path: Path) -> DataPreparationResult:
    prepared_path = tmp_path / "cleaned.csv"
    prepared_path.write_text("date,sales,profit,region\n2024-01-01,10,2,A\n", encoding="utf-8")
    return DataPreparationResult(
        output_path=prepared_path.as_posix(),
        operations=["drop_duplicates"],
        row_count=1,
        col_count=4,
    )


@pytest.fixture
def visrag_result() -> VisRAGResult:
    return VisRAGResult(
        recommendations=[
            VisRAGRecommendation(
                chart_family="line",
                rationale="Time-like field detected; line chart suits trend analysis.",
                priority=1,
            )
        ],
        visualization_plan=VisualizationPlan(
            chart_family="line",
            visual_task="trend_analysis",
            goal="Show sales trend over time",
            title="Show sales trend over time",
            field_bindings=[
                VisualizationFieldBinding(channel="x", field_name="date", field_role="temporal", title="Date"),
                VisualizationFieldBinding(channel="y", field_name="sales", field_role="quantitative", title="Sales", aggregate="mean"),
            ],
            axes=[
                VisualizationAxisInstruction(channel="x", field_name="date", title="Date", scale_type="temporal", rotate_labels=True),
                VisualizationAxisInstruction(channel="y", field_name="sales", title="Sales", scale_type="linear"),
            ],
            build_instructions=[
                "Use the temporal field on the x-axis.",
                "Use the sales measure on the y-axis.",
            ],
            renderer_hints=["Prefer readable defaults."],
            confidence=0.9,
        ),
        rules=["Use clear titles and axis labels."],
        caveats=[],
    )


@pytest.fixture
def code_run_result(tmp_path: Path) -> CodeRunResult:
    plot_path = tmp_path / "plot.png"
    plot_path.write_bytes(b"png")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    metadata_path = tmp_path / "chart_metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    return CodeRunResult(
        success=True,
        metrics=[
            ExecutionMetric(name="row_count", value=4),
            ExecutionMetric(name="column_count", value=3),
            ExecutionMetric(name="y_min", value=10),
            ExecutionMetric(name="y_max", value=18),
            ExecutionMetric(name="y_mean", value=13.75),
        ],
        artifacts=[
            ArtifactRef(
                artifact_type=ArtifactType.PLOT,
                path=plot_path.as_posix(),
                description="Generated chart image.",
            ),
            ArtifactRef(
                artifact_type=ArtifactType.METRICS,
                path=metrics_path.as_posix(),
                description="Execution metrics.",
            ),
            ArtifactRef(
                artifact_type=ArtifactType.CHART_METADATA,
                path=metadata_path.as_posix(),
                description="Chart metadata.",
            ),
        ],
        chart_metadata={
            "chart_type": "line",
            "title": "Sales trend",
            "x_column": "date",
            "y_column": "sales",
        },
    )


@pytest.fixture
def chart_read_result(tmp_path: Path) -> ChartReadResult:
    plot_path = tmp_path / "plot.png"
    plot_path.write_bytes(b"png")
    return ChartReadResult(
        chart_type="line",
        title="Sales trend",
        axes={"x": "date", "y": "sales"},
        series=["sales"],
        source_artifact=plot_path.as_posix(),
    )


@pytest.fixture
def facts_result() -> FactExtractionResult:
    evidence = ["/tmp/plot.png"]
    return FactExtractionResult(
        facts=[
            Fact(name="chart_type", value="line", evidence=evidence),
            Fact(name="y_min", value="10", evidence=evidence),
            Fact(name="y_max", value="18", evidence=evidence),
            Fact(name="y_mean", value="13.75", evidence=evidence),
        ]
    )


@pytest.fixture
def reasoning_result() -> ReasoningResult:
    return ReasoningResult(
        summary="The selected chart type is line.",
        statements=[
            ReasoningStatement(
                text="The selected chart type is line.",
                evidence=["/tmp/plot.png"],
            )
        ],
    )
