from __future__ import annotations

from pathlib import Path

from src.domain.models import (
    CandidateSpec,
    CandidateSpecSet,
    DataProfile,
    QueryUnderstandingResult,
    RequestAnalysisResult,
    VisualizationAxisInstruction,
    VisualizationFieldBinding,
    VisualizationPlan,
    VisRAGRecommendation,
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
        domain_examples = [self._to_domain_example(item) for item in result.candidates]
        recommendations = [self._to_recommendation(index, item) for index, item in enumerate(result.candidates, start=1)]
        candidate_set = self._to_candidate_set(result.candidates, domain_examples, query_understanding)
        return VisRAGResult(
            recommendations=recommendations,
            visualization_plan=candidate_set.visualization_plan,
            rules=[],
            caveats=[],
            implementation_notes=candidate_set.visualization_plan.build_instructions if candidate_set.visualization_plan else [],
            retrieved_examples=domain_examples,
            corpus_status={"root": result.corpus_root, "examples": str(len(domain_examples))},
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
        query_parts = [query_understanding.intent, query_understanding.user_goal or "", query_understanding.analysis_goal or ""]
        query_parts.extend(variant.text for variant in query_understanding.query_variants)
        query_parts.extend(request_analysis.selected_fields)
        return VisRAGRequest(
            query=" ".join(part for part in query_parts if part).strip(),
            data_profile=VisRAGDataProfile(
                columns=[
                    VisRAGColumnProfile(name=column.name, semantic_type=column.dtype, role=data_profile.field_roles.get(column.name))
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
    def _to_recommendation(priority: int, candidate: CoreCandidate) -> VisRAGRecommendation:
        return VisRAGRecommendation(
            chart_family=candidate.example.chart_type,
            rationale="Matched prepared RAG corpus example.",
            priority=priority,
            score=candidate.score,
            support_examples=[candidate.example.example_id],
            instruction_highlights=[candidate.example.instruction],
        )

    def _to_candidate_set(
        self,
        candidates: list[CoreCandidate],
        examples: list[VisRAGRetrievedExample],
        query_understanding: QueryUnderstandingResult,
    ) -> CandidateSpecSet:
        specs = [self._to_candidate_spec(item, query_understanding) for item in candidates]
        selected = specs[0] if specs else None
        return CandidateSpecSet(
            candidate_specs=specs,
            retrieved_examples=examples,
            visualization_plan=selected.visualization_plan if selected else None,
            ranking_hints=[],
            selected_candidate_spec=selected,
        )

    @staticmethod
    def _to_candidate_spec(candidate: CoreCandidate, query_understanding: QueryUnderstandingResult) -> CandidateSpec:
        example = candidate.example
        plan = _build_plan(candidate, query_understanding)
        return CandidateSpec(
            spec_id=example.example_id,
            chart_family=example.chart_type,
            summary=example.description or example.instruction,
            score=candidate.score,
            rationale="Matched prepared RAG corpus example.",
            visualization_plan=plan,
            spec_template=candidate.spec_template,
            encoding_roles=dict(example.field_roles),
            transform_types=list(example.transform_types),
        )


def _build_plan(candidate: CoreCandidate, query_understanding: QueryUnderstandingResult) -> VisualizationPlan:
    chart_type = candidate.example.chart_type
    bindings = [
        VisualizationFieldBinding(
            channel=channel,
            field_name=field_name,
            field_role=_vega_type(candidate.example.field_roles.get(channel)),
            title=field_name,
            aggregate=_channel_aggregate(candidate.spec_template, channel),
        )
        for channel, field_name in candidate.field_mapping.items()
    ]
    return VisualizationPlan(
        chart_family=chart_type,
        visual_task=query_understanding.task_type or chart_type,
        goal=query_understanding.user_goal or query_understanding.intent,
        title=_title(query_understanding.intent),
        description=candidate.example.description or candidate.example.instruction,
        field_bindings=bindings,
        transforms=[],
        axes=[
            VisualizationAxisInstruction(
                channel=binding.channel,
                field_name=binding.field_name,
                title=binding.title or binding.field_name,
                scale_type=binding.field_role,
                format_hint="%Y-%m-%d" if binding.field_role == "temporal" else None,
                rotate_labels=binding.channel == "x",
            )
            for binding in bindings
            if binding.channel in {"x", "y"}
        ],
        build_instructions=["Generate Vega-Lite from the selected prepared-corpus candidate."],
        evidence_example_ids=[candidate.example.example_id],
        confidence=candidate.score,
    )


def _vega_type(value: str | None) -> str:
    text = (value or "nominal").lower()
    if text in {"numeric", "number", "measure", "quantitative"}:
        return "quantitative"
    if text in {"date", "time", "datetime", "temporal", "year"}:
        return "temporal"
    return "nominal"


def _channel_aggregate(spec: dict, channel: str) -> str | None:
    encoding = spec.get("encoding") if isinstance(spec, dict) else None
    channel_spec = encoding.get(channel) if isinstance(encoding, dict) else None
    aggregate = channel_spec.get("aggregate") if isinstance(channel_spec, dict) else None
    return str(aggregate) if aggregate else None


def _title(text: str) -> str:
    return (text or "Generated chart").strip().capitalize()
