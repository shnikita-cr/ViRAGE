from __future__ import annotations

from pathlib import Path

from src.domain.models import (
    CandidateSpec,
    CandidateSpecSet,
    DataColumnProfile,
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
    semantic_type_from_role_or_dtype,
)
from src.visrag_core.grounding_policy import resolve_grounding_policy
from src.visrag_core.models import VisRAGCandidate as CoreCandidate


class VisRAGService(BaseService):
    """ViRAGE adapter around the portable VisRAG core."""

    def invoke(
            self,
            query_understanding: QueryUnderstandingResult,
            request_analysis: RequestAnalysisResult,
            data_profile: DataProfile,
            runtime: RuntimeContext,
    ) -> VisRAGResult:
        if not bool(getattr(runtime.settings, "visrag_enabled", True)):
            empty_set = CandidateSpecSet(candidate_specs=[], retrieved_examples=[],
                                         ranking_hints=["VisRAG disabled by project settings."])
            return VisRAGResult(
                caveats=["VisRAG disabled by project settings."],
                retrieved_examples=[],
                corpus_status={"enabled": "false", "examples": "0"},
                retrieval_strategy="disabled",
                retrieval_query="",
                candidate_spec_set=empty_set,
            )

        request = self._to_core_request(query_understanding, request_analysis, data_profile, runtime)
        core = VisRAGCoreService(
            VisRAGConfig(
                corpus_root=self._corpus_root(runtime),
                feedback_corpus_path=getattr(runtime.settings, "semantic_feedback_corpus_path", None),
                include_feedback_corpus=bool(getattr(runtime.settings, "visrag_include_feedback_corpus", True)),
                retriever_backend=getattr(runtime.settings, "visrag_retriever_backend", "bm25"),
                embedding_provider=getattr(runtime.settings, "visrag_embedding_provider", None),
                embedding_model=getattr(runtime.settings, "visrag_embedding_model", None),
                embedding_base_url=getattr(runtime.settings, "visrag_embedding_base_url", None),
                embedding_timeout_seconds=getattr(runtime.settings, "visrag_embedding_timeout_seconds", 60.0),
            )
        )
        result = core.search(request)
        examples = [self._to_domain_example(candidate) for candidate in result.candidates]
        candidate_specs = [self._to_candidate_spec(candidate) for candidate in result.candidates]
        candidate_set = CandidateSpecSet(candidate_specs=candidate_specs, retrieved_examples=examples)
        candidate_set.selected_candidate_spec = candidate_specs[0] if candidate_specs else None
        return VisRAGResult(
            caveats=list(result.caveats),
            retrieved_examples=examples,
            corpus_status={"root": result.corpus_root, "examples": str(len(examples))},
            retrieval_strategy=f"prepared_corpus:{getattr(runtime.settings, 'visrag_retriever_backend', 'bm25')}",
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
        request = VisRAGRequest(
            query=self._search_query(query_understanding, request_analysis),
            data_profile=VisRAGDataProfile(
                columns=[self._to_core_column(column, data_profile) for column in data_profile.columns]),
            preferred_chart_types=[canonicalize_chart_type(item) for item in query_understanding.candidate_charts],
            selected_fields=list(request_analysis.selected_fields),
            selected_fields_policy="auto",
            top_k=runtime.settings.visrag_top_k_examples,
        )
        policy_decision = resolve_grounding_policy(
            request,
            request_confidence=self._request_confidence(query_understanding, request_analysis),
            ambiguity_notes=self._ambiguity_notes(query_understanding, request_analysis),
        )
        return request.model_copy(
            update={"selected_fields_policy": policy_decision.selected_fields_policy.value}
        )

    @staticmethod
    def _request_confidence(
            query_understanding: QueryUnderstandingResult,
            request_analysis: RequestAnalysisResult,
    ) -> float | None:
        if request_analysis.confidence > 0:
            return request_analysis.confidence
        if query_understanding.confidence > 0:
            return query_understanding.confidence
        return None

    @staticmethod
    def _ambiguity_notes(
            query_understanding: QueryUnderstandingResult,
            request_analysis: RequestAnalysisResult,
    ) -> list[str]:
        return [
            *list(query_understanding.ambiguity_notes),
            *list(request_analysis.ambiguity_report),
        ]

    @staticmethod
    def _search_query(query_understanding: QueryUnderstandingResult, request_analysis: RequestAnalysisResult) -> str:
        query_parts = [
            query_understanding.intent,
            query_understanding.user_goal or "",
            query_understanding.analysis_goal or "",
            *[variant.text for variant in query_understanding.query_variants],
            *request_analysis.selected_fields,
        ]
        return " ".join(part for part in query_parts if part).strip()

    @staticmethod
    def _to_core_column(column: DataColumnProfile, data_profile: DataProfile) -> VisRAGColumnProfile:
        role = data_profile.field_roles.get(column.name)
        semantic_type = semantic_type_from_role_or_dtype(role, column.dtype, column.dtype)
        return VisRAGColumnProfile(
            name=column.name,
            semantic_type=semantic_type,
            role=role,
            raw_dtype=column.dtype,
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
            metadata={
                "confidence": candidate.confidence,
                "field_mapping": candidate.field_mapping,
                "score_breakdown": candidate.score_breakdown,
                **example.metadata,
            },
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
