from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.domain.models import (
    ChartAnswerJudgeResult,
    ChartFactSummaryResult,
    QueryRequestAnalysisResult,
    VLMChartDescriptionResult,
    VegaLiteSpecArtifact,
    VisualFeedbackExample,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class FeedbackCorpusWriterService(BaseService):
    def invoke(self, *args, **kwargs):
        raise NotImplementedError("Use build_example() and append_to_corpus() for explicit graph-node writes.")

    def build_example(
            self,
            *,
            run_id: str,
            attempt_number: int,
            query: str,
            vega_spec: VegaLiteSpecArtifact,
            rendered_png_path: str,
            vlm_description: VLMChartDescriptionResult,
            chart_facts: ChartFactSummaryResult,
            judge_result: ChartAnswerJudgeResult,
            request_analysis: QueryRequestAnalysisResult | None = None,
    ) -> VisualFeedbackExample:
        return VisualFeedbackExample(
            status="accepted" if judge_result.retry_recommendation == "accept" else "rejected_or_needs_improvement",
            created_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            attempt_number=attempt_number,
            user_query=query,
            request_analysis_summary=request_analysis.model_dump() if request_analysis is not None else {},
            generated_spec=vega_spec.spec_json,
            rendered_png_path=rendered_png_path,
            vlm_chart_description=vlm_description,
            chart_fact_summary=chart_facts,
            judge_result=judge_result,
            feedback_for_next_generation=judge_result.feedback_for_next_generation,
            feedback_weight=1.5 if judge_result.feedback_for_next_generation else 1.0,
            rag_usage={
                "approved_for_rag": False,
                "exported_to_rag": False,
                "approved_for_retrieval": False,
                "source_quality": "vlm_structured_feedback",
                "weight": 1.5 if judge_result.feedback_for_next_generation else 1.0,
            },
        )

    def build_user_feedback_example(
            self,
            *,
            run_id: str,
            query: str,
            comment: str,
            needs_regeneration: bool,
            vega_spec: VegaLiteSpecArtifact,
            rendered_png_path: str,
            request_analysis: QueryRequestAnalysisResult | None = None,
            attempt_number: int = 1,
    ) -> VisualFeedbackExample:
        cleaned_comment = comment.strip()
        judge_result = ChartAnswerJudgeResult(
            answers_user_query=not needs_regeneration,
            confidence=1.0,
            retry_recommendation="retry" if needs_regeneration else "accept",
            missing_requirements=[] if not needs_regeneration else [cleaned_comment],
            wrong_or_suspicious_parts=[],
            improvement_comments=[cleaned_comment] if cleaned_comment else [],
            feedback_for_next_generation=cleaned_comment if needs_regeneration else "",
        )
        return VisualFeedbackExample(
            source="virage_user_feedback",
            status="rejected_or_needs_improvement" if needs_regeneration else "accepted",
            created_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            attempt_number=attempt_number,
            user_query=query,
            user_comment=cleaned_comment,
            requested_regeneration=needs_regeneration,
            feedback_weight=3.0 if needs_regeneration else 2.0,
            request_analysis_summary=request_analysis.model_dump() if request_analysis is not None else {},
            generated_spec=vega_spec.spec_json,
            rendered_png_path=rendered_png_path,
            vlm_chart_description=VLMChartDescriptionResult(
                visual_description="Manual user feedback was provided after final chart rendering.",
                confidence=1.0,
            ),
            chart_fact_summary=ChartFactSummaryResult(
                facts=[cleaned_comment] if cleaned_comment else [],
                quality_notes=["manual_user_feedback"],
            ),
            judge_result=judge_result,
            feedback_for_next_generation=cleaned_comment if needs_regeneration else "",
            rag_usage={
                "approved_for_rag": False,
                "exported_to_rag": False,
                "approved_for_retrieval": False,
                "saved_as_feedback_log": True,
                "priority": "manual_review_required",
                "weight": 3.0 if needs_regeneration else 2.0,
            },
        )

    def append_to_feedback_log(self, example: VisualFeedbackExample, runtime: RuntimeContext) -> str:
        """Append feedback to a JSONL log only.

        The log is not a runtime RAG corpus. A separate reviewed export step must
        decide which feedback records are allowed to enter retrieval.
        """
        path = Path(runtime.settings.semantic_feedback_corpus_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = example.model_dump()
        payload["rag_usage"] = {
            **(payload.get("rag_usage") or {}),
            "approved_for_rag": False,
            "exported_to_rag": False,
            "approved_for_retrieval": False,
            "saved_as_feedback_log": True,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
        return path.as_posix()

    def append_to_corpus(self, example: VisualFeedbackExample, runtime: RuntimeContext) -> str:
        return self.append_to_feedback_log(example, runtime)
