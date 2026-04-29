from __future__ import annotations

from pathlib import Path

from src.domain.models import (
    CandidateSpec,
    CandidateSpecSet,
    DataProfile,
    QueryUnderstandingResult,
    RequestAnalysisResult,
    VisRAGResult,
    VisRAGRetrievedExample,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.visrag_core import (
    VisRAGColumnProfile,
    VisRAGConfig,
    VisRAGCoreService,
    VisRAGDataProfile,
    VisRAGRequest,
    canonicalize_chart_type,
)
from src.visrag_core.models import VisRAGCandidate as CoreCandidate


class VisRAGService(BaseService):
    """Thin ViRAGE adapter around the portable VisRAG core."""

    def invoke(
            self,
            query_understanding: QueryUnderstandingResult,
            request_analysis: RequestAnalysisResult,
            data_profile: DataProfile,
            runtime: RuntimeContext,
    ) -> VisRAGResult:
        request = self._to_core_request(query_understanding, request_analysis, data_profile, runtime)
        core = VisRAGCoreService(VisRAGConfig(corpus_root=self._corpus_root(runtime)))
        result = core.search(request)
        examples = [self._to_domain_example(candidate) for candidate in result.candidates]
        candidate_set = CandidateSpecSet(
            candidate_specs=[self._to_candidate_spec(candidate) for candidate in result.candidates],
            retrieved_examples=examples,
        )
        candidate_set.selected_candidate_spec = candidate_set.candidate_specs[
            0] if candidate_set.candidate_specs else None
        return VisRAGResult(
            retrieved_examples=examples,
            corpus_status={"root": result.corpus_root, "examples": str(len(examples))},
            retrieval_strategy="prepared_corpus",
            retrieval_query=request.query,
            candidate_spec_set=candidate_set,
        )

    def _to_core_request(
            self,
            query_understanding: QueryUnderstandingResult,
            request_analysis: RequestAnalysisResult,
            data_profile: DataProfile,
            runtime: RuntimeContext,
    ) -> VisRAGRequest:
        query_parts = [
            query_understanding.intent,
            query_understanding.user_goal or "",
            query_understanding.analysis_goal or "",
            *[variant.text for variant in query_understanding.query_variants],
            *request_analysis.selected_fields,
        ]
        return VisRAGRequest(
            query=" ".join(part for part in query_parts if part).strip(),
            data_profile=VisRAGDataProfile(
                columns=[
                    VisRAGColumnProfile(
                        name=column.name,
                        semantic_type=column.dtype,
                        role=data_profile.field_roles.get(column.name),
                    )
                    for column in data_profile.columns
                ]
            ),
            preferred_chart_types=[canonicalize_chart_type(item) for item in query_understanding.candidate_charts],
            selected_fields=list(request_analysis.selected_fields),
            top_k=runtime.settings.visrag_top_k_examples,
        )

    @staticmethod
    def _corpus_root(runtime: RuntimeContext) -> Path:
        return runtime.settings.visrag_corpus_root or Path("rag_corpus/data")

    @staticmethod
    def _to_domain_example(candidate: CoreCandidate) -> VisRAGRetrievedExample:
        example = candidate.example
        return VisRAGRetrievedExample(
            example_id=example.example_id,
            source=example.source,
            corpus=example.corpus,
            chart_type=example.chart_type,
            instruction=example.instruction,
            description=example.description,
            tags=example.keywords,
            score=candidate.score,
            rationale="prepared corpus match",
            metadata={"field_mapping": candidate.field_mapping, **example.metadata},
        )

    @staticmethod
    def _to_candidate_spec(candidate: CoreCandidate) -> CandidateSpec:
        example = candidate.example
        return CandidateSpec(
            spec_id=example.example_id,
            chart_family=example.chart_type,
            summary=example.description or example.instruction,
            score=candidate.score,
            rationale="Matched prepared RAG corpus example.",
            spec_template=candidate.spec_template,
            encoding_roles=dict(example.field_roles),
            transform_types=list(example.transform_types),
            support_examples=[example.example_id],
            field_mapping=dict(candidate.field_mapping),
        )
