from __future__ import annotations

import json
from pathlib import Path

from src.application.settings import ViRAGESettings
from src.domain.models import (
    ChartAnswerJudgeResult,
    ChartFactSummaryResult,
    DataPreparationResult,
    SemanticFeedbackLoopSummary,
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
        candidate_spec_set={"candidate_specs": [], "retrieved_examples": []},  # type: ignore[arg-type]
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
