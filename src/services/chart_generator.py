from __future__ import annotations

from src.domain.models import (
    CandidateSpecSet,
    DataPreparationResult,
    DataProfile,
    QueryUnderstandingResult,
    RequestAnalysisResult,
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
            request_analysis: RequestAnalysisResult | None = None,
            query_understanding: QueryUnderstandingResult | None = None,
            visrag: VisRAGResult | None = None,
    ) -> VegaLiteSpecArtifact:
        if candidate_spec_set.selected_candidate_spec is None and not candidate_spec_set.candidate_specs:
            raise RuntimeError("Chart generation requires at least one RAG candidate or retrieved context.")

        request = SpecGenerationRequest(
            query=query,
            prepared=prepared,
            candidate_spec_set=candidate_spec_set,
            data_profile=data_profile,
            request_analysis=request_analysis,
            query_understanding=query_understanding,
            visrag=visrag,
        )
        backend = self._select_backend(runtime)
        result = backend.generate(request, runtime)
        return VegaLiteSpecArtifact(
            spec_json=result.spec_json,
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
            request_analysis: RequestAnalysisResult | None = None,
            query_understanding: QueryUnderstandingResult | None = None,
            visrag: VisRAGResult | None = None,
    ) -> VegaLiteSpecArtifact:
        return self.invoke(
            prepared=prepared,
            candidate_spec_set=candidate_spec_set,
            runtime=runtime,
            query=query,
            data_profile=data_profile,
            request_analysis=request_analysis,
            query_understanding=query_understanding,
            visrag=visrag,
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
