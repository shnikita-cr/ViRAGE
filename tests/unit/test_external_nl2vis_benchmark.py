from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.benchmark.external.adapters import (
    DataFormulatorCommandAdapter,
    DataFormulatorHttpAdapter,
    ExternalAdapterError,
    NL4DVAdapter,
    extract_vegalite_spec,
)
from src.benchmark.external.ollama import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    OllamaSettings,
    assert_ollama_model_available,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


def test_extract_vegalite_spec_from_nl4dv_vis_list() -> None:
    spec = {"mark": "bar", "encoding": {"x": {"field": "a"}}}
    payload = {"visList": [{"vlSpec": spec}]}

    assert extract_vegalite_spec(payload) == spec


def test_ollama_settings_defaults_to_local_gemma3_4b() -> None:
    settings = OllamaSettings()

    assert settings.host == DEFAULT_OLLAMA_HOST
    assert settings.model == DEFAULT_OLLAMA_MODEL == "gemma3:4b"
    assert settings.as_litellm_config() == {
        "model": "ollama/gemma3:4b",
        "api_key": "ollama",
        "api_base": "http://localhost:11434",
    }


def test_nl4dv_ollama_adapter_uses_language_model_mode() -> None:
    adapter = NL4DVAdapter.with_ollama(settings=OllamaSettings())

    assert adapter.system_name == "nl4dv_ollama"
    assert adapter.processing_mode == "language-model"
    assert adapter.lm_config == {
        "model": "ollama/gemma3:4b",
        "api_key": "ollama",
        "api_base": "http://localhost:11434",
    }


def test_data_formulator_http_ollama_payload_contains_llm_contract(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("category,value\nA,1\n", encoding="utf-8")
    adapter = DataFormulatorHttpAdapter.with_ollama(
        endpoint="http://localhost:5567/benchmark/generate",
        settings=OllamaSettings(),
        include_data_records=True,
        max_records=200,
    )

    payload = adapter._payload(query="show value by category", data_path=data_path, case_id="case-a")

    assert payload["case_id"] == "case-a"
    assert payload["query"] == "show value by category"
    assert payload["data_path"] == data_path.as_posix()
    assert payload["llm"] == {
        "provider": "ollama",
        "model": "gemma3:4b",
        "api_base": "http://localhost:11434",
    }
    assert payload["columns"] == ["category", "value"]
    assert payload["records"] == [{"category": "A", "value": 1}]


def test_assert_ollama_model_available_accepts_exact_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse({"models": [{"name": "gemma3:4b"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert_ollama_model_available(OllamaSettings(), timeout_seconds=1.0)


def test_assert_ollama_model_available_rejects_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse({"models": [{"name": "qwen3.5:4b"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(ExternalAdapterError, match="gemma3:4b"):
        assert_ollama_model_available(OllamaSettings(), timeout_seconds=1.0)


def test_data_formulator_command_adapter_reads_output_json(tmp_path: Path) -> None:
    script = tmp_path / "adapter.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[sys.argv.index('--input') + 1])
output_path = Path(sys.argv[sys.argv.index('--output') + 1])
payload = json.loads(input_path.read_text(encoding='utf-8'))
output_path.write_text(json.dumps({
    'generated_spec': {
        'mark': 'bar',
        'encoding': {'x': {'field': 'category'}, 'y': {'field': 'value'}},
    },
    'case_id': payload['case_id'],
}), encoding='utf-8')
""".strip(),
        encoding="utf-8",
    )
    data_path = tmp_path / "data.csv"
    data_path.write_text("category,value\nA,1\n", encoding="utf-8")
    adapter = DataFormulatorCommandAdapter(
        command_template=f"python {script.as_posix()} --input {{input_json}} --output {{output_json}}",
    )

    result = adapter.generate(query="show value by category", data_path=data_path, case_id="case-a")

    assert result.generated_spec["mark"] == "bar"
    assert result.raw_output["case_id"] == "case-a"
