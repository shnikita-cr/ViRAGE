from src.application.contracts import PipelineResult
from src.application.settings import ViRAGESettings
from src.domain.enums import ChartCaseType, PipelineStage
from src.domain.models import (
    AnalysisRubric,
    CandidateSpec,
    CandidateSpecSet,
    ExecutionPolicy,
    PlanningResult,
    QueryUnderstandingResult,
    ValidationPolicy,
    VisualizationPlan,
)
from src.infrastructure.runtime import RuntimeContext


def test_query_understanding_supports_new_intent_bundle_fields() -> None:
    result = QueryUnderstandingResult(
        intent="Show sales trend",
        requested_operations=["trend analysis"],
        candidate_charts=["line"],
        constraints=[],
        case_type=ChartCaseType.CANONICAL,
        confidence=0.9,
        task_type="trend",
        user_goal="understand change over time",
        analysis_goal="identify trend shifts",
        ambiguity_notes=["time granularity not specified"],
    )

    bundle = result.to_intent_bundle()

    assert bundle.intent == result.intent
    assert bundle.analysis_goal == "identify trend shifts"
    assert bundle.ambiguity_notes == ["time granularity not specified"]


def test_planning_result_supports_policy_objects() -> None:
    planning = PlanningResult(
        execution_policy=ExecutionPolicy(max_retries=2, fallback_enabled=True),
        validation_policy=ValidationPolicy(use_spec_validator=True, use_empty_chart_check=True),
        analysis_rubric=AnalysisRubric(focus_areas=["trend", "anomaly"], strict_visual_only=True),
    )

    assert planning.execution_policy is not None
    assert planning.validation_policy is not None
    assert planning.analysis_rubric is not None


def test_candidate_spec_set_can_hold_ranked_specs() -> None:
    plan = VisualizationPlan(
        chart_family="line",
        visual_task="trend_analysis",
        goal="Show sales trend over time",
        title="Sales trend",
    )
    spec = CandidateSpec(
        spec_id="cand-1",
        chart_family="line",
        summary="Line chart for temporal trend",
        score=0.9,
        visualization_plan=plan,
    )
    candidate_set = CandidateSpecSet(
        candidate_specs=[spec],
        visualization_plan=plan,
        selected_candidate_spec=spec,
    )

    assert candidate_set.selected_candidate_spec is not None
    assert candidate_set.selected_candidate_spec.spec_id == "cand-1"


def test_runtime_context_supports_new_model_slots(tmp_path) -> None:
    runtime = RuntimeContext(
        settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"),
        reasoning_llm=object(),
        spec_llm=object(),
        vlm=object(),
        vision_judge_llm=object(),
    )

    assert runtime.reasoning_llm is not None
    assert runtime.spec_llm is not None
    assert runtime.vlm is not None
    assert runtime.vision_judge_llm is not None


def test_pipeline_result_allows_new_optional_artifacts() -> None:
    result_fields = PipelineResult.model_fields

    assert "request_analysis" in result_fields
    assert "candidate_spec_set" in result_fields
    assert "vega_spec" in result_fields
    assert "evaluation_summary" in result_fields


def test_pipeline_stage_supports_new_architecture_stages() -> None:
    assert PipelineStage.REQUEST_ANALYSIS.value == "request_analysis"
    assert PipelineStage.SPEC_VALIDATION.value == "spec_validation"
    assert PipelineStage.VLM_ANALYSIS.value == "vlm_analysis"
