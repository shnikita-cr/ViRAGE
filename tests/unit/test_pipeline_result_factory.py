from __future__ import annotations

from src.application.results.pipeline_result_factory import PipelineResultFactory
from src.application.runtime.state import PipelineState
from src.domain.models import InsightsResult, TokenUsage


class _RuntimeStub:
    def token_usage_summary(self) -> TokenUsage:
        return TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)


def test_pipeline_result_factory_keeps_public_optional_fields() -> None:
    state: PipelineState = {
        "run_id": "run_1",
        "query": "show chart",
        "data_path": "data.csv",
        "insights": InsightsResult(final_insights=["visible trend"]),
        "artifact_paths": {"plot": "plot.png"},
        "step_logs": [],
        "stage_execution_logs": [],
        "model_call_logs": [],
    }

    result = PipelineResultFactory.from_state(state, _RuntimeStub())

    assert result.run_id == "run_1"
    assert result.insights is not None
    assert result.insights.final_insights == ["visible trend"]
    assert result.artifact_paths == {"plot": "plot.png"}
    assert result.token_usage_summary.total_tokens == 3
