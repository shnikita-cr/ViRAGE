from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.benchmark.external.data_formulator_payload import (
    add_records_to_payload,
    base_data_formulator_payload,
)
from src.benchmark.external.errors import ExternalAdapterError
from src.benchmark.external.ollama import OllamaSettings


@dataclass(slots=True)
class ExternalAdapterResult:
    generated_spec: dict[str, Any]
    raw_output: dict[str, Any] = field(default_factory=dict)
    stdout: str | None = None
    stderr: str | None = None


class ExternalNL2VISAdapter(Protocol):
    system_name: str

    def generate(self, *, query: str, data_path: Path, case_id: str) -> ExternalAdapterResult:
        ...


@dataclass(slots=True)
class NL4DVAdapter:
    processing_mode: str = "semantic-parsing"
    dependency_parser_config: dict[str, Any] | None = None
    lm_config: dict[str, Any] | None = None
    gpt_api_key: str | None = None
    verbose: bool = False
    system_name_value: str = "nl4dv"

    @property
    def system_name(self) -> str:
        return self.system_name_value

    @classmethod
    def with_ollama(
        cls,
        *,
        settings: OllamaSettings,
        dependency_parser_config: dict[str, Any] | None = None,
        verbose: bool = False,
        system_name_value: str = "nl4dv_ollama",
    ) -> "NL4DVAdapter":
        return cls(
            processing_mode="language-model",
            dependency_parser_config=dependency_parser_config,
            lm_config=settings.as_litellm_config(),
            verbose=verbose,
            system_name_value=system_name_value,
        )

    def generate(self, *, query: str, data_path: Path, case_id: str) -> ExternalAdapterResult:
        try:
            from nl4dv import NL4DV  # type: ignore
        except ImportError as exc:
            raise ExternalAdapterError("NL4DV is not installed. Install it in the active environment first.") from exc

        kwargs: dict[str, Any] = {
            "data_url": data_path.as_posix(),
            "processing_mode": self.processing_mode,
        }
        if self.lm_config is not None:
            kwargs["lm_config"] = self.lm_config
        if self.gpt_api_key:
            kwargs["gpt_api_key"] = self.gpt_api_key
        instance = NL4DV(**kwargs)
        if self.dependency_parser_config is not None:
            instance.set_dependency_parser(config=self.dependency_parser_config)
        raw_output = instance.analyze_query(query, verbose=self.verbose)
        if not isinstance(raw_output, dict):
            raise ExternalAdapterError(f"NL4DV returned {type(raw_output).__name__}, expected dict.")
        spec = extract_vegalite_spec(raw_output)
        if not spec:
            raise ExternalAdapterError("NL4DV output does not contain a Vega-Lite specification.")
        return ExternalAdapterResult(generated_spec=spec, raw_output=raw_output)


@dataclass(slots=True)
class DataFormulatorHttpAdapter:
    endpoint: str
    timeout_seconds: float = 180.0
    include_data_records: bool = False
    max_records: int = 200
    llm: dict[str, Any] | None = None
    system_name_value: str = "data_formulator"

    @property
    def system_name(self) -> str:
        return self.system_name_value

    @classmethod
    def with_ollama(
        cls,
        *,
        endpoint: str,
        settings: OllamaSettings,
        timeout_seconds: float = 180.0,
        include_data_records: bool = False,
        max_records: int = 200,
        system_name_value: str = "data_formulator_http_ollama",
    ) -> "DataFormulatorHttpAdapter":
        return cls(
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            include_data_records=include_data_records,
            max_records=max_records,
            llm=settings.as_payload(),
            system_name_value=system_name_value,
        )

    def generate(self, *, query: str, data_path: Path, case_id: str) -> ExternalAdapterResult:
        payload = self._payload(query=query, data_path=data_path, case_id=case_id)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise ExternalAdapterError(f"Data Formulator HTTP request failed: {exc}") from exc
        try:
            raw_output = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExternalAdapterError(f"Data Formulator HTTP response is not valid JSON: {exc}") from exc
        if not isinstance(raw_output, dict):
            raise ExternalAdapterError("Data Formulator HTTP response must be a JSON object.")
        spec = extract_vegalite_spec(raw_output)
        if not spec:
            raise ExternalAdapterError("Data Formulator HTTP response does not contain a Vega-Lite specification.")
        return ExternalAdapterResult(generated_spec=spec, raw_output=raw_output)

    def _payload(self, *, query: str, data_path: Path, case_id: str) -> dict[str, Any]:
        payload = base_data_formulator_payload(query=query, data_path=data_path, case_id=case_id, llm=self.llm)
        add_records_to_payload(
            payload,
            data_path=data_path,
            include_data_records=self.include_data_records,
            max_records=self.max_records,
        )
        return payload


@dataclass(slots=True)
class DataFormulatorCommandAdapter:
    command_template: str
    timeout_seconds: float = 300.0
    include_data_records: bool = False
    max_records: int = 200
    llm: dict[str, Any] | None = None
    system_name_value: str = "data_formulator"

    @property
    def system_name(self) -> str:
        return self.system_name_value

    @classmethod
    def with_ollama(
        cls,
        *,
        command_template: str,
        settings: OllamaSettings,
        timeout_seconds: float = 300.0,
        include_data_records: bool = False,
        max_records: int = 200,
        system_name_value: str = "data_formulator_command_ollama",
    ) -> "DataFormulatorCommandAdapter":
        return cls(
            command_template=command_template,
            timeout_seconds=timeout_seconds,
            include_data_records=include_data_records,
            max_records=max_records,
            llm=settings.as_payload(),
            system_name_value=system_name_value,
        )

    def generate(self, *, query: str, data_path: Path, case_id: str) -> ExternalAdapterResult:
        with tempfile.TemporaryDirectory(prefix="virage_df_case_") as temp_dir_raw:
            temp_dir = Path(temp_dir_raw)
            input_path = temp_dir / "input.json"
            output_path = temp_dir / "output.json"
            input_path.write_text(
                json.dumps(self._payload(query=query, data_path=data_path, case_id=case_id), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            command = self._command(input_path=input_path, output_path=output_path, query=query, data_path=data_path)
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise ExternalAdapterError(
                    f"Data Formulator command failed with exit code {completed.returncode}: {completed.stderr.strip()}"
                )
            raw_output = _read_command_output(output_path=output_path, stdout=completed.stdout)
            spec = extract_vegalite_spec(raw_output)
            if not spec:
                raise ExternalAdapterError("Data Formulator command output does not contain a Vega-Lite specification.")
            return ExternalAdapterResult(
                generated_spec=spec,
                raw_output=raw_output,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    def _payload(self, *, query: str, data_path: Path, case_id: str) -> dict[str, Any]:
        payload = base_data_formulator_payload(query=query, data_path=data_path, case_id=case_id, llm=self.llm)
        add_records_to_payload(
            payload,
            data_path=data_path,
            include_data_records=self.include_data_records,
            max_records=self.max_records,
        )
        return payload

    def _command(self, *, input_path: Path, output_path: Path, query: str, data_path: Path) -> list[str]:
        command = self.command_template.format(
            input_json=input_path.as_posix(),
            output_json=output_path.as_posix(),
            query=shlex.quote(query),
            data_path=data_path.as_posix(),
        )
        return shlex.split(command)


def extract_vegalite_spec(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("vlSpec", "vl_spec", "vega_lite_spec", "spec", "generated_spec"):
        value = payload.get(key)
        if isinstance(value, dict) and _looks_like_vegalite(value):
            return value
    design = payload.get("vlSpec_design")
    if isinstance(design, dict) and _looks_like_vegalite(design):
        return design
    vis_list = payload.get("visList")
    if isinstance(vis_list, list):
        for item in vis_list:
            if not isinstance(item, dict):
                continue
            for key in ("vlSpec", "vl_spec", "vega_lite_spec", "spec"):
                value = item.get(key)
                if isinstance(value, dict) and _looks_like_vegalite(value):
                    return value
    charts = payload.get("charts") or payload.get("visualizations")
    if isinstance(charts, list):
        for item in charts:
            if isinstance(item, dict):
                spec = extract_vegalite_spec(item)
                if spec:
                    return spec
    return {}


def _looks_like_vegalite(value: dict[str, Any]) -> bool:
    return "mark" in value or "encoding" in value or "layer" in value or "hconcat" in value or "vconcat" in value


def _read_command_output(*, output_path: Path, stdout: str) -> dict[str, Any]:
    raw = output_path.read_text(encoding="utf-8") if output_path.exists() else stdout
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExternalAdapterError(f"External command did not produce valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ExternalAdapterError("External command output must be a JSON object.")
    return parsed

