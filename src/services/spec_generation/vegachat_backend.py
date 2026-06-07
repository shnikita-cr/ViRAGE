from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.domain.models import SpecGenerationAttempt, SpecGenerationRequest, SpecGenerationResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_text
from src.services.spec_generation.base import SpecGenerationBackend
from src.services.spec_generation.vegachat_parser import VegaChatResponseParseError, parse_vegachat_response
from src.services.spec_generation.vegachat_prompts import VEGA_LITE_SCHEMA_URL, build_vegachat_codegen_prompt


class VegaChatCodegenBackend(SpecGenerationBackend):
    """VegaChat-style LLM/codegen backend for Vega-Lite specs.

    This backend intentionally does not run VegaChat's UI/session layer. It reuses
    the VegaChat prompt contract (<explain>/<json>) inside the ViRAGE runtime, so
    model calls, stages and artifacts remain in one ViRAGE run directory.
    """

    backend_name = "vegachat_codegen"

    def generate(self, request: SpecGenerationRequest, runtime: RuntimeContext) -> SpecGenerationResult:
        if runtime.spec_llm is None:
            raise RuntimeError("VegaChat codegen backend requires RuntimeContext.spec_llm.")

        prompt_version = runtime.settings.spec_generation_prompt_version
        attempts, warning_messages, explanation, spec_without_data = self._run_generation_attempts(
            request=request,
            runtime=runtime,
            prompt_version=prompt_version,
            max_attempts=max(1, int(runtime.settings.spec_generation_max_attempts)),
            max_context_chars=int(runtime.settings.spec_generation_max_context_chars),
        )
        final_spec = self._attach_runtime_data(spec_without_data, request.prepared.output_path)
        result = SpecGenerationResult(
            backend_name=self.backend_name,
            prompt_version=prompt_version,
            spec_json=final_spec,
            spec_without_runtime_data=spec_without_data,
            explanation=explanation,
            attempts=attempts,
            warning_messages=list(dict.fromkeys(warning_messages)),
            used_visrag_context=self._uses_visrag_context(request, runtime),
            generation_attempt_number=request.generation_attempt_number,
            max_generation_attempts=request.max_generation_attempts,
            previous_validation_errors=list(request.previous_validation_errors),
            previous_repair_hints=list(request.previous_repair_hints),
            previous_semantic_feedback=list(request.previous_semantic_feedback),
        )
        return result.model_copy(update={"artifact_paths": {}})

    def _run_generation_attempts(
            self,
            *,
            request: SpecGenerationRequest,
            runtime: RuntimeContext,
            prompt_version: str,
            max_attempts: int,
            max_context_chars: int,
    ) -> tuple[list[SpecGenerationAttempt], list[str], str | None, dict[str, Any]]:
        attempts: list[SpecGenerationAttempt] = []
        warning_messages: list[str] = []
        previous_error: str | None = None
        previous_response: str | None = None
        final_raw_response = ""

        for attempt_number in range(1, max_attempts + 1):
            prompt = self._build_prompt(
                request=request,
                runtime=runtime,
                prompt_version=prompt_version,
                max_context_chars=max_context_chars,
                previous_error=previous_error,
                previous_response=previous_response,
            )
            try:
                raw_response = invoke_text(
                    runtime.spec_llm,
                    prompt,
                    runtime=runtime,
                    stage="chart_generator",
                    role="spec",
                )
                final_raw_response = raw_response
                explanation, parsed_spec = parse_vegachat_response(raw_response)
                spec_without_data, policy_warnings = self._normalize_model_spec(parsed_spec)
                warning_messages.extend(policy_warnings)
                attempts.append(
                    SpecGenerationAttempt(
                        attempt_number=attempt_number,
                        status="succeeded",
                        raw_response=raw_response,
                        explanation=explanation,
                        spec_json=spec_without_data,
                    )
                )
                return attempts, warning_messages, explanation, spec_without_data
            except (VegaChatResponseParseError, ValueError, TypeError) as exc:
                previous_error = f"{type(exc).__name__}: {exc}"
                previous_response = final_raw_response
                attempts.append(
                    SpecGenerationAttempt(
                        attempt_number=attempt_number,
                        status="failed",
                        raw_response=final_raw_response,
                        error=previous_error,
                    )
                )
                if attempt_number >= max_attempts:
                    raise RuntimeError(
                        f"VegaChat codegen failed after {max_attempts} attempts. Last error: {previous_error}"
                    ) from exc

        raise RuntimeError("VegaChat codegen did not produce a Vega-Lite spec.")

    @staticmethod
    def _build_prompt(
            *,
            request: SpecGenerationRequest,
            runtime: RuntimeContext,
            prompt_version: str,
            max_context_chars: int,
            previous_error: str | None,
            previous_response: str | None,
    ) -> str:
        return build_vegachat_codegen_prompt(
            request,
            prompt_version=prompt_version,
            max_context_chars=max_context_chars,
            include_visrag_context=bool(runtime.settings.spec_generation_include_visrag_context),
            previous_error=previous_error,
            previous_response=previous_response,
            rag_prompt_top_k=int(getattr(runtime.settings, "visrag_prompt_top_k_examples", 2)),
        )

    @staticmethod
    def _uses_visrag_context(request: SpecGenerationRequest, runtime: RuntimeContext) -> bool:
        return bool(
            runtime.settings.spec_generation_include_visrag_context
            and request.visrag is not None
            and request.visrag.generation_guidance.has_guidance
        )

    @staticmethod
    def _normalize_model_spec(spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        clone = deepcopy(spec)
        warnings: list[str] = []
        if "data" in clone:
            clone.pop("data", None)
            warnings.append("model_output_data_removed_before_runtime_attachment")
        if "datasets" in clone:
            clone.pop("datasets", None)
            warnings.append("model_output_datasets_removed_before_runtime_attachment")
        if "$schema" not in clone:
            clone["$schema"] = VEGA_LITE_SCHEMA_URL
            warnings.append("model_output_missing_schema_added")
        return clone, warnings

    @staticmethod
    def _attach_runtime_data(spec_without_data: dict[str, Any], output_path: str) -> dict[str, Any]:
        spec = deepcopy(spec_without_data)
        spec["data"] = {"url": output_path}
        return spec

    @staticmethod
    def _save_artifacts(
            *,
            runtime: RuntimeContext,
            result: SpecGenerationResult,
            final_prompt: str,
            final_raw_response: str,
            generation_attempt_number: int,
    ) -> dict[str, str]:
        artifact_paths: dict[str, str] = {}
        if not runtime.current_run_id:
            return artifact_paths

        attempt_prefix = f"spec_generation_attempt_{generation_attempt_number:03d}"

        artifact_paths[f"{attempt_prefix}_prompt"] = runtime.save_text_artifact(
            f"nodes/{attempt_prefix}_prompt.txt",
            final_prompt,
            numbered=True,
        )
        artifact_paths[f"{attempt_prefix}_raw_response"] = runtime.save_text_artifact(
            f"nodes/{attempt_prefix}_raw_response.txt",
            final_raw_response,
            numbered=True,
        )
        artifact_paths[f"{attempt_prefix}_parsed_response"] = runtime.save_json_artifact(
            f"nodes/{attempt_prefix}_parsed_response.json",
            {
                "explanation": result.explanation,
                "spec_without_runtime_data": result.spec_without_runtime_data,
                "warnings": result.warning_messages,
            },
            numbered=True,
        )
        result_payload = result.model_dump()
        result_payload["artifact_paths"] = dict(artifact_paths)
        artifact_paths[f"{attempt_prefix}_result"] = runtime.save_json_artifact(
            f"nodes/{attempt_prefix}_result.json",
            result_payload,
            numbered=True,
        )
        result_payload["artifact_paths"] = dict(artifact_paths)
        return artifact_paths
