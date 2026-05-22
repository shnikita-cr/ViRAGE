from __future__ import annotations

import json
from pathlib import Path

from src.application.settings import ViRAGESettings
from src.domain.models import (
    ChartAnswerJudgeResult,
    ChartFactSummaryResult,
    DataPreparationResult,
    VLMChartDescriptionResult,
    VegaLiteSpecArtifact,
)
from src.domain.spec_generation_models import SpecGenerationRequest
from src.infrastructure.runtime import RuntimeContext
from src.services.spec_generation.vegachat_prompts import build_vegachat_codegen_prompt
from src.services.visual_feedback.feedback_corpus_writer import FeedbackCorpusWriterService


def test_semantic_feedback_is_injected_into_spec_generation_prompt() -> None:
    prepared = DataPreparationResult(
        output_path="prepared.csv",
        row_count=10,
        col_count=2,
        original_columns=["Method", "PSNR"],
        safe_columns=["Method", "PSNR"],
    )
    request = SpecGenerationRequest(
        query="compare PSNR by Method",
        prepared=prepared,
        previous_semantic_feedback=["Previous chart did not show PSNR grouped by Method."],
        previous_chart_facts=[{"facts": ["The chart shows counts, not PSNR."]}],
    )

    prompt = build_vegachat_codegen_prompt(
        request,
        prompt_version="test",
        max_context_chars=1000,
        include_visrag_context=False,
    )

    assert "Previous rendered chart was technically valid" in prompt
    assert "Previous chart did not show PSNR" in prompt
    assert "The chart shows counts" in prompt


def test_feedback_corpus_writer_appends_visual_feedback_jsonl(tmp_path: Path) -> None:
    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
            semantic_feedback_corpus_path=tmp_path / "visual_feedback.jsonl",
        )
    )
    writer = FeedbackCorpusWriterService()
    example = writer.build_example(
        run_id="run-1",
        attempt_number=1,
        query="compare PSNR by Method",
        vega_spec=VegaLiteSpecArtifact(spec_json={"mark": "bar"}),
        rendered_png_path="plot.png",
        vlm_description=VLMChartDescriptionResult(visual_description="A bar chart is visible."),
        chart_facts=ChartFactSummaryResult(facts=["The chart compares bar heights."]),
        judge_result=ChartAnswerJudgeResult(
            answers_user_query=False,
            confidence=0.4,
            missing_requirements=["PSNR is not visible."],
            improvement_comments=["Use PSNR on the quantitative axis."],
            feedback_for_next_generation="Use PSNR by Method.",
        ),
    )
    path = writer.append_to_corpus(example, runtime)

    payload = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])
    assert payload["record_type"] == "visual_feedback"
    assert payload["rag_usage"]["approved_for_rag"] is False
    assert payload["feedback_for_next_generation"] == "Use PSNR by Method."


def test_chart_answer_judge_accepts_common_no_retry_tokens() -> None:
    from src.services.visual_feedback.chart_answer_judge import _normalize_retry_recommendation

    for token in ["no_retry", "none", "ok", "accepted", "do not retry"]:
        assert _normalize_retry_recommendation(
            token,
            answers_user_query=True,
            feedback_for_next_generation="",
            missing_requirements=[],
            wrong_or_suspicious_parts=[],
            improvement_comments=[],
        ) == "accept"


def test_chart_answer_judge_keeps_retry_when_actionable_feedback_exists() -> None:
    from src.services.visual_feedback.chart_answer_judge import _normalize_retry_recommendation

    assert _normalize_retry_recommendation(
        "retry",
        answers_user_query=True,
        feedback_for_next_generation="Rotate X labels.",
        missing_requirements=[],
        wrong_or_suspicious_parts=[],
        improvement_comments=[],
    ) == "retry"


def test_feedback_corpus_writer_appends_user_feedback_with_high_weight(tmp_path: Path) -> None:
    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
            semantic_feedback_corpus_path=tmp_path / "visual_feedback.jsonl",
        )
    )
    writer = FeedbackCorpusWriterService()
    example = writer.build_user_feedback_example(
        run_id="run-user",
        attempt_number=1,
        query="compare methods",
        comment="Use a horizontal bar chart with readable labels.",
        needs_regeneration=True,
        vega_spec=VegaLiteSpecArtifact(spec_json={"mark": "bar"}),
        rendered_png_path="plot.png",
    )
    path = writer.append_to_corpus(example, runtime)

    payload = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])
    assert payload["source"] == "virage_user_feedback"
    assert payload["requested_regeneration"] is True
    assert payload["feedback_weight"] == 3.0
    assert payload["rag_usage"]["approved_for_rag"] is True
    assert payload["feedback_for_next_generation"] == "Use a horizontal bar chart with readable labels."


def test_semantic_chart_judge_prompt_is_png_only_and_omits_spec_data() -> None:
    from src.services.visual_feedback.semantic_chart_judge import SemanticChartJudgeService

    prompt = SemanticChartJudgeService._prompt(
        query="compare values",
        request_analysis=None,
        visual_judge_requirements={
            "must_be_visible": ["The x and y fields are visible."],
            "yes_no_questions": ["Are the requested values visible in the chart image?"],
        },
        use_chartsquared=True,
        chartsquared_max_eval_questions=8,
        prompt_max_chars=6000,
        chartsquared_project_root=None,
    )

    assert "PNG-only judge payload" in prompt
    assert "compare values" in prompt
    assert "generated_vega_lite_spec_without_runtime_data" not in prompt
    assert "source table" in prompt
    assert "Are the requested values visible" in prompt
