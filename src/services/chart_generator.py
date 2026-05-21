from __future__ import annotations

from src.domain.models import (
    CandidateSpecSet,
    DataPreparationResult,
    DataProfile,
    QueryRequestAnalysisResult,
    SpecGenerationRequest,
    VegaLiteSpecArtifact,
    VisRAGResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.spec_generation import TemplateSpecBackend, VegaChatCodegenBackend
from src.services.spec_generation.base import SpecGenerationBackend


class ChartGeneratorService(BaseService):
    """Generate a Vega-Lite artifact through a configurable spec backend."""

    def invoke(
            self,
            prepared: DataPreparationResult,
            candidate_spec_set: CandidateSpecSet,
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
        if (
                backend.backend_name == "template"
                and candidate_spec_set.selected_candidate_spec is None
                and not candidate_spec_set.candidate_specs
        ):
            raise RuntimeError("Template chart generation requires at least one RAG candidate.")

        request = SpecGenerationRequest(
            query=query,
            prepared=prepared,
            candidate_spec_set=candidate_spec_set,
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
        return VegaLiteSpecArtifact(
            spec_json=result.spec_json,
            spec_without_runtime_data=result.spec_without_runtime_data,
            version="v2" if result.backend_name == "vegachat_codegen" else "v1",
            generation_backend=result.backend_name,
            generation_explanation=result.explanation,
            generation_warnings=list(result.warning_messages),
            generation_artifacts=dict(result.artifact_paths),
        )

    async def ainvoke(
            self,
            prepared: DataPreparationResult,
            candidate_spec_set: CandidateSpecSet,
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
        return self.invoke(
            prepared=prepared,
            candidate_spec_set=candidate_spec_set,
            runtime=runtime,
            query=query,
            data_profile=data_profile,
            query_request_analysis=query_request_analysis,
            visrag=visrag,
            generation_attempt_number=generation_attempt_number,
            max_generation_attempts=max_generation_attempts,
            previous_validation_errors=previous_validation_errors,
            previous_repair_hints=previous_repair_hints,
            previous_invalid_spec=previous_invalid_spec,
            previous_semantic_feedback=previous_semantic_feedback,
            previous_chart_facts=previous_chart_facts,
            visual_judge_requirements=visual_judge_requirements,
        )

    @staticmethod
    def _select_backend(runtime: RuntimeContext) -> SpecGenerationBackend:
        backend_name = str(runtime.settings.spec_generation_backend).strip().lower()
        if backend_name in {"template", "legacy_template", "materializer"}:
            return TemplateSpecBackend()
        if backend_name in {"vegachat_codegen", "vegachat", "llm_codegen"}:
            return VegaChatCodegenBackend()
        raise ValueError(
            "Unsupported spec_generation_backend: "
            f"{runtime.settings.spec_generation_backend!r}. "
            "Expected 'template' or 'vegachat_codegen'."
        )
