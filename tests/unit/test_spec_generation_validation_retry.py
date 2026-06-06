from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.application.config.settings import ViRAGESettings
from src.domain.models import (
    DataPreparationResult,
    SpecValidationResult,
    VegaLiteSpecArtifact,
)
from src.graph.nodes import PipelineNodes
from src.infrastructure.runtime import RuntimeContext


@dataclass
class FakeLLMResult:
    content: str
    usage_metadata: dict[str, int]


class FakeSpecLLM:
    model = "fake-spec-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, messages: Any) -> FakeLLMResult:
        self.prompts.append(str(messages))
        return FakeLLMResult(
            content=self.response,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


class FakeSpecValidator:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, vega_spec: VegaLiteSpecArtifact) -> SpecValidationResult:
        self.calls += 1
        if self.calls == 1:
            return SpecValidationResult(
                validated_spec=vega_spec.spec_json,
                validation_errors=["Unsupported mark type: invalid_mark."],
                repair_hints=["Use a supported mark type such as bar."],
                is_valid=False,
            )
        return SpecValidationResult(
            validated_spec=vega_spec.spec_json,
            validation_errors=[],
            repair_hints=[],
            is_valid=True,
        )


def _base_state() -> dict[str, Any]:
    prepared = DataPreparationResult(
        output_path="prepared.csv",
        row_count=2,
        col_count=2,
        original_columns=["Category", "Value"],
        safe_columns=["Category", "Value"],
    )
    return {
        "run_id": "run",
        "query": "compare value by category",
        "data_preparation": prepared,
        "vega_spec": VegaLiteSpecArtifact(
            spec_json={
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {"url": "prepared.csv"},
                "mark": "invalid_mark",
                "encoding": {},
            },
            generation_backend="vegachat_codegen",
        ),
        "artifact_paths": {},
        "step_logs": [],
        "trace": [],
        "technical_attempt_number": 1,
    }


def test_graph_level_technical_retry_passes_validation_feedback_to_next_generation(tmp_path: Path) -> None:
    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
            spec_generation_backend="vegachat_codegen",
            spec_generation_max_attempts=2,
            spec_generation_response_parse_retries=0,
        ),
        spec_llm=FakeSpecLLM(
            "<explain>Fixed invalid mark.</explain>"
            "<json>{\"$schema\":\"https://vega.github.io/schema/vega-lite/v5.json\","
            "\"mark\":\"bar\","
            "\"encoding\":{\"x\":{\"field\":\"Category\",\"type\":\"nominal\"},"
            "\"y\":{\"field\":\"Value\",\"type\":\"quantitative\"}}}</json>"
        ),
    )
    runtime.current_run_id = "run"
    runtime.ensure_run_dir()

    nodes = PipelineNodes(runtime)
    nodes.spec_validator = FakeSpecValidator()
    state = _base_state()

    validation_output = nodes.spec_validator_node(state)
    state.update(validation_output)
    assert state["spec_validation"].is_valid is False

    decision_output = nodes.technical_decision_node(state)
    state.update(decision_output)
    assert state["technical_status"] == "retry"
    assert state["technical_attempt_number"] == 2

    generation_output = nodes.chart_generator_node(state)
    state.update(generation_output)
    assert state["vega_spec"].spec_json["mark"] == "bar"
    assert "Unsupported mark type" in runtime.spec_llm.prompts[0]

    validation_output_2 = nodes.spec_validator_node(state)
    assert validation_output_2["spec_validation"].is_valid is True
    assert nodes.spec_validator.calls == 2

    run_nodes = tmp_path / "artifacts" / "run" / "nodes"
    assert list(run_nodes.glob("*vega_spec*.json"))
    assert list(run_nodes.glob("*spec_validation*.json"))
    assert list(run_nodes.glob("*technical_decision*.json"))
