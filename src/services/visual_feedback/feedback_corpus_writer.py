from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.domain.models import (
    ChartAnswerJudgeResult,
    ChartFactSummaryResult,
    RequestAnalysisResult,
    SemanticFeedbackLoopSummary,
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
        request_analysis: RequestAnalysisResult | None = None,
    ) -> VisualFeedbackExample:
        return VisualFeedbackExample(
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
        )

    def append_to_corpus(self, example: VisualFeedbackExample, runtime: RuntimeContext) -> str:
        path = Path(runtime.settings.semantic_feedback_corpus_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(example.model_dump(), ensure_ascii=False, default=str) + "\n")
        return path.as_posix()
