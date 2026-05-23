from __future__ import annotations

from src.domain.models import (
    DataPreparationResult,
    DataProfile,
    QueryRequestAnalysisResult,
    SpecGenerationRequest,
    VegaLiteSpecArtifact,
    VisRAGResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.spec_generation import VegaChatCodegenBackend
from src.services.spec_presentation_consistency import SpecPresentationConsistencyService
from src.services.spec_generation.base import SpecGenerationBackend


class ChartGeneratorService(BaseService):
    """Generate a Vega-Lite artifact through the internal spec backend."""

    def invoke(
            self,
            prepared: DataPreparationResult,
            runtime: RuntimeContext,
            *,
            query: str = "",
            data_profile: DataProfile | None = None,
            query_request_analysis: QueryRequestAnalysisResult | None = None,
            visrag: VisRAGResult | None = None,
            generation_attempt_number: int = 1,
            max_generation_attempts: int = 1,
            previous_validation_errors: list[str] | None = None,
            previous_repair_hints: list[str] | None = None,
            previous_invalid_spec: dict | None = None,
            previous_semantic_feedback: list[str] | None = None,
            previous_chart_facts: list[dict] | None = None,
            visual_judge_requirements: dict | None = None,
    ) -> VegaLiteSpecArtifact:
        backend = self._select_backend(runtime)
        request = SpecGenerationRequest(
            query=query,
            prepared=prepared,
            data_profile=data_profile,
            query_request_analysis=query_request_analysis,
            visrag=visrag,
            generation_attempt_number=generation_attempt_number,
            max_generation_attempts=max_generation_attempts,
            previous_validation_errors=list(previous_validation_errors or []),
            previous_repair_hints=list(previous_repair_hints or []),
            previous_invalid_spec=previous_invalid_spec,
            previous_semantic_feedback=list(previous_semantic_feedback or []),
            previous_chart_facts=list(previous_chart_facts or []),
            visual_judge_requirements=dict(visual_judge_requirements or {}),
        )
        result = backend.generate(request, runtime)
        presentation = SpecPresentationConsistencyService()
        spec_json_result = presentation.normalize(result.spec_json)
        spec_without_data_result = presentation.normalize(result.spec_without_runtime_data)
        presentation_notes = [
            *spec_json_result.changes,
            *(note for note in spec_without_data_result.changes if note not in spec_json_result.changes),
        ]
        return VegaLiteSpecArtifact(
            spec_json=spec_json_result.spec,
            spec_without_runtime_data=spec_without_data_result.spec,
            version="v2",
            generation_backend=result.backend_name,
            generation_explanation=result.explanation,
            generation_warnings=[*result.warning_messages, *presentation_notes],
            generation_artifacts=dict(result.artifact_paths),
        )

    async def ainvoke(
            self,
            prepared: DataPreparationResult,
            runtime: RuntimeContext,
            **kwargs,
    ) -> VegaLiteSpecArtifact:
        return self.invoke(prepared=prepared, runtime=runtime, **kwargs)

    @staticmethod
    def _select_backend(runtime: RuntimeContext) -> SpecGenerationBackend:
        backend_name = str(runtime.settings.spec_generation_backend).strip().lower()
        if backend_name in {"vegachat_codegen", "vegachat", "llm_codegen"}:
            return VegaChatCodegenBackend()
        raise ValueError(
            "Unsupported spec_generation_backend: "
            f"{runtime.settings.spec_generation_backend!r}. Expected 'vegachat_codegen'."
        )
