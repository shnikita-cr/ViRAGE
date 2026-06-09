from __future__ import annotations

import json
import sys
import types
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
        "environ_var_name": "OLLAMA_API_KEY",
        "api_key": "ollama",
        "api_base": "http://localhost:11434",
    }


def test_nl4dv_ollama_adapter_uses_language_model_mode() -> None:
    adapter = NL4DVAdapter.with_ollama(settings=OllamaSettings())

    assert adapter.system_name == "nl4dv_ollama"
    assert adapter.processing_mode == "language-model"
    assert adapter.lm_config == {
        "model": "ollama/gemma3:4b",
        "environ_var_name": "OLLAMA_API_KEY",
        "api_key": "ollama",
        "api_base": "http://localhost:11434",
    }


def test_nl4dv_ollama_adapter_binds_configured_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("category,value\nA,1\n", encoding="utf-8")

    class FakeNL4DV:
        used_models: list[str] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def analyze_query(self, query: str, verbose: bool = False) -> dict[str, Any]:
            return self.query_language_model([{"type": "text", "text": query}])

        def query_language_model(self, prompts: Any, model: str = "gpt-4o-mini") -> dict[str, Any]:
            FakeNL4DV.used_models.append(model)
            return {"vlSpec": {"mark": "bar", "encoding": {"x": {"field": "category"}}}}

    fake_module = types.ModuleType("nl4dv")
    fake_module.NL4DV = FakeNL4DV
    monkeypatch.setitem(sys.modules, "nl4dv", fake_module)

    adapter = NL4DVAdapter.with_ollama(settings=OllamaSettings())
    result = adapter.generate(query="show value by category", data_path=data_path, case_id="case-a")

    assert FakeNL4DV.used_models == ["ollama/gemma3:4b"]
    assert result.generated_spec["mark"] == "bar"


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


def test_external_runner_passes_runtime_to_evaluator(tmp_path: Path) -> None:
    from src.application.config.settings import ViRAGESettings
    from src.benchmark.core.models import BenchmarkCase, BenchmarkCaseResult
    from src.benchmark.external.adapters import ExternalAdapterResult
    from src.benchmark.external.runner import ExternalNL2VISBenchmarkRunner
    from src.infrastructure.runtime import RuntimeContext

    data_path = tmp_path / "data.csv"
    data_path.write_text("category,value\nA,1\n", encoding="utf-8")
    runtime = RuntimeContext(settings=ViRAGESettings(), vlm=object())
    observed: dict[str, object] = {}

    class FakeAdapter:
        system_name = "fake_external"

        def generate(self, *, query: str, data_path: Path, case_id: str) -> ExternalAdapterResult:
            return ExternalAdapterResult(
                generated_spec={"mark": "bar", "encoding": {"x": {"field": "category"}}},
                raw_output={"ok": True},
            )

    class FakeEvaluator:
        def render_reference_image(self, *, reference_spec: dict[str, Any], data_path: str, output_path: Path) -> str:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"not-a-real-png")
            return output_path.as_posix()

        def evaluate_spec_and_image(self, **kwargs: Any) -> BenchmarkCaseResult:
            observed["runtime"] = kwargs["runtime"]
            case = kwargs["case"]
            return BenchmarkCaseResult(
                case_id=case.case_id,
                dataset_name=case.dataset_name,
                query=case.query,
                data_path=case.resolved_data_path(kwargs["case_root"]),
                generated_spec=kwargs["generated_spec"],
                generated_image_path=kwargs["generated_image_path"],
                is_valid_spec=True,
                spec_score=1.0,
                vision_score=0.5,
            )

    runner = ExternalNL2VISBenchmarkRunner(adapter=FakeAdapter(), evaluator=FakeEvaluator(), runtime=runtime)
    case = BenchmarkCase(case_id="case-a", query="show value", data_path="data.csv", dataset_name="test")

    result = runner.run_case(case=case, case_root=tmp_path, output_dir=tmp_path / "out")

    assert result.error is None
    assert observed["runtime"] is runtime
    assert result.vision_score == 0.5


def test_vision_runtime_loads_vlm_from_toml_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from src.application.config.settings import ViRAGESettings
    from scripts.benchmark.runners.external import run_nvbench20_external_nl2vis_benchmark as runner_script

    settings = ViRAGESettings()
    vlm_config = SimpleNamespace(model="gemma3:4b")
    built_model = object()

    def fake_load_project_config(path: Path) -> SimpleNamespace:
        return SimpleNamespace(mode="pipeline", settings=settings, vlm_model=vlm_config)

    def fake_build_chat_model(config: object, model_settings: object, *, role: str | None = None) -> object:
        assert config is vlm_config
        assert model_settings is not settings
        assert role == "vlm"
        return built_model

    monkeypatch.setattr(runner_script, "load_project_config", fake_load_project_config)
    monkeypatch.setattr(runner_script, "build_chat_model", fake_build_chat_model)

    runtime = runner_script._vision_runtime(SimpleNamespace(vision_config="config.toml"))

    assert runtime is not None
    assert runtime.vlm is built_model
    assert runtime.settings is not settings


def test_vision_runtime_is_optional() -> None:
    from types import SimpleNamespace

    from scripts.benchmark.runners.external import run_nvbench20_external_nl2vis_benchmark as runner_script

    assert runner_script._vision_runtime(SimpleNamespace(vision_config=None)) is None
