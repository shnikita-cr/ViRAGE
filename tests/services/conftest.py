from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.settings import ViRAGESettings
from src.domain.models import (
    AnalysisRubric,
    CandidateSpec,
    CandidateSpecSet,
    DataPreparationResult,
    DataProfile,
    PlotImageArtifact,
    QueryUnderstandingResult,
    RequestAnalysisResult,
    ScenegraphCheckResult,
    VisualizationAxisInstruction,
    VisualizationFieldBinding,
    VisualizationPlan,
    VisRAGRecommendation,
    VisRAGResult,
    VLMAnalysisResult,
    VisualFact,
    VisualFactExtractionResult,
    InsightCandidate,
    InsightReasoningResult,
)
from src.infrastructure.runtime import RuntimeContext
from tests._fakes import FakeReasoningLLM, FakeSpecLLM, FakeVLM, FakeVisionJudgeLLM


@pytest.fixture
def runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        settings=ViRAGESettings(artifact_root=tmp_path / "artifacts", visrag_enable_llm_synthesis=False),
        reasoning_llm=FakeReasoningLLM(),
        spec_llm=FakeSpecLLM(),
        vlm=FakeVLM(),
        vision_judge_llm=FakeVisionJudgeLLM(),
    )


@pytest.fixture
def canonical_query_understanding() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        intent="Show sales trend over time",
        requested_operations=["trend analysis"],
        candidate_charts=["line", "bar"],
        constraints=[],
        confidence=0.9,
        task_type="trend_analysis",
        user_goal="understand sales movement over time",
        analysis_goal="find trend shifts and peaks",
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
        field_roles={"date": "temporal", "sales": "quantitative", "region": "nominal"},
        schema_hints=["date is temporal", "sales is quantitative"],
    )


@pytest.fixture
def prepared_result(tmp_path: Path) -> DataPreparationResult:
    prepared_path = tmp_path / "cleaned.csv"
    prepared_path.write_text("date,sales,profit,region\n2024-01-01,10,2,A\n2024-01-02,14,3,B\n", encoding="utf-8")
    return DataPreparationResult(
        output_path=prepared_path.as_posix(),
        operations=["drop_duplicates"],
        row_count=2,
        col_count=4,
    )


@pytest.fixture
def request_analysis() -> RequestAnalysisResult:
    return RequestAnalysisResult(
        grounded_fields=["date", "sales"],
        selected_fields=["date", "sales", "region"],
        ambiguity_report=[],
        normalization_hints=["parse date as datetime"],
        confidence=0.95,
    )


@pytest.fixture
def visrag_result() -> VisRAGResult:
    plan = VisualizationPlan(
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
    )
    selected = CandidateSpec(spec_id="candidate_1_line", chart_family="line", summary="Top candidate", score=0.9, rationale="Top candidate", visualization_plan=plan)
    return VisRAGResult(
        recommendations=[
            VisRAGRecommendation(
                chart_family="line",
                rationale="Time-like field detected; line chart suits trend analysis.",
                priority=1,
            )
        ],
        visualization_plan=plan,
        candidate_spec_set=CandidateSpecSet(
            candidate_specs=[selected],
            visualization_plan=plan,
            ranking_hints=["prefer line"],
            selected_candidate_spec=selected,
        ),
        rules=["Use clear titles and axis labels."],
        caveats=[],
    )


@pytest.fixture
def analysis_rubric() -> AnalysisRubric:
    return AnalysisRubric(focus_areas=["trend", "peaks", "anomalies"], output_format="bullet_points", strict_visual_only=True, emphasize_anomalies=True)


@pytest.fixture
def plot_image(tmp_path: Path) -> PlotImageArtifact:
    import matplotlib.pyplot as plt
    image_path = tmp_path / "plot.png"
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [2, 4, 5])
    fig.savefig(image_path)
    plt.close(fig)
    return PlotImageArtifact(image_path=image_path.as_posix(), width=640, height=480)


@pytest.fixture
def vlm_analysis_result() -> VLMAnalysisResult:
    return VLMAnalysisResult(
        visual_observations=["The line trends upward.", "There is a local peak near the end."],
        extracted_visual_facts=["The chart shows an increasing temporal trend.", "A peak appears near the final portion of the line."],
        confidence=0.85,
    )


@pytest.fixture
def visual_facts_result() -> VisualFactExtractionResult:
    return VisualFactExtractionResult(
        visual_facts=[
            VisualFact(name="visual_fact_1", value="The chart shows an increasing temporal trend.", evidence_refs=["observation:1"]),
            VisualFact(name="visual_fact_2", value="A peak appears near the final portion of the line.", evidence_refs=["observation:2"]),
        ],
        evidence_refs=["observation:1", "observation:2"],
    )


@pytest.fixture
def insight_reasoning_result() -> InsightReasoningResult:
    return InsightReasoningResult(
        insight_candidates=[
            InsightCandidate(statement="Sales increase over time with a late peak.", confidence=0.87, reasoning_chain=["The line rises from left to right.", "A noticeable high point appears near the end."])
        ],
        reasoning_chain=["Observation 1 indicates upward movement.", "Observation 2 indicates a late peak."],
    )


@pytest.fixture
def scenegraph_status() -> ScenegraphCheckResult:
    return ScenegraphCheckResult(has_marks=True, has_axes=True, has_legends=False, notes=[])
