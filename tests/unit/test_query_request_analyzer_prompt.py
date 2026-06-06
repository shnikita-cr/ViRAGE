from __future__ import annotations

from src.application.config.settings import ViRAGESettings
from src.domain.data.data_models import DataColumnProfile, DataProfile
from src.infrastructure.runtime import RuntimeContext
from src.llm.model_runtime import attach_runtime_profile, resolve_model_runtime_profile
from src.services.planning.query_request_analyzer import QueryRequestAnalyzerService


class _FakeModel:
    pass


def test_query_request_prompt_accepts_runtime_context() -> None:
    model = attach_runtime_profile(
        _FakeModel(),
        resolve_model_runtime_profile(
            provider="ollama",
            model_name="gemma3:4b",
            role="reasoning",
            gpu_ram_gb=8.0,
        ),
    )
    runtime = RuntimeContext(settings=ViRAGESettings(), reasoning_llm=model)
    data_profile = DataProfile(
        row_count=2,
        col_count=2,
        columns=[
            DataColumnProfile(name="metric", dtype="float", role="measure", sample_values=[1.0, 2.0]),
            DataColumnProfile(name="group", dtype="string", role="dimension", sample_values=["A", "B"]),
        ],
    )

    prompt = QueryRequestAnalyzerService()._prompt("compare metric by group", {}, data_profile, runtime)

    assert "compare metric by group" in prompt
    assert "metric" in prompt
    assert "group" in prompt


def test_query_request_schema_accepts_field_binding_only_payload() -> None:
    from src.services.planning.query_request_analyzer import _QueryRequestAnalysisSchema

    parsed = _QueryRequestAnalysisSchema.model_validate(
        {
            "Method": {"field": "Method", "confidence": 0.65},
            "score": {"field": "Score", "role": "measure", "confidence": 0.7},
            "confidence": 0.65,
        }
    )

    assert parsed.normalized_query == ""
    assert parsed.selected_fields == ["Method", "Score"]
    assert parsed.field_bindings["Method"].field == "Method"
    assert parsed.field_bindings["score"].field == "Score"


def test_query_request_schema_rejects_empty_object() -> None:
    import pytest
    from pydantic import ValidationError
    from src.services.planning.query_request_analyzer import _QueryRequestAnalysisSchema

    with pytest.raises(ValidationError):
        _QueryRequestAnalysisSchema.model_validate({"confidence": 0.65})
