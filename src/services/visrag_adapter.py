from __future__ import annotations

from src.domain.models import (
    CandidateSpec,
    CandidateSpecSet,
    DataProfile,
    QueryUnderstandingResult,
    RequestAnalysisResult,
    VisualizationAxisInstruction,
    VisualizationFieldBinding,
    VisualizationPlan,
    VisualizationTransform,
    VisRAGRecommendation,
    VisRAGResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.visrag_core import (
    VisRAGColumnProfile,
    VisRAGConfig,
    VisRAGCoreService,
    VisRAGDataProfile,
    VisRAGExample,
    VisRAGRequest,
)
from src.visrag_core.models import VisRAGCandidate as CoreCandidate


class VisRAGService(BaseService):
    """ViRAGE adapter for the portable VisRAG core.

    The portable core receives plain DTOs and returns plain DTOs. This adapter is
    the only place where ViRAGE domain/runtime models are converted to and from
    the core API.
    """

    def invoke(
            self,
            query_understanding: QueryUnderstandingResult,
            request_analysis: RequestAnalysisResult,
            data_profile: DataProfile,
            runtime: RuntimeContext,
    ) -> VisRAGResult:
        core_request = self._to_core_request(query_understanding, request_analysis, data_profile, runtime)
        core_config = self._to_core_config(runtime)
        core_result = VisRAGCoreService(core_config).search(core_request)
        return self._to_domain_result(core_result, query_understanding, request_analysis, data_profile)

    @staticmethod
    def _to_core_config(runtime: RuntimeContext) -> VisRAGConfig:
        settings = runtime.settings
        return VisRAGConfig(
            corpus_root=settings.visrag_corpus_root,
            index_root=settings.visrag_index_root,
            force_rebuild_index=settings.visrag_force_rebuild_index,
            retriever_fetch_k=settings.visrag_retriever_fetch_k,
            similarity_threshold=settings.visrag_similarity_threshold,
            embedding_backend=settings.visrag_embedding_backend,
            embedding_model=settings.visrag_embedding_model,
            ollama_base_url=settings.visrag_ollama_base_url,
            ollama_timeout_seconds=settings.visrag_ollama_timeout_seconds,
        )

    @classmethod
    def _to_core_request(
            cls,
            query_understanding: QueryUnderstandingResult,
            request_analysis: RequestAnalysisResult,
            data_profile: DataProfile,
            runtime: RuntimeContext,
    ) -> VisRAGRequest:
        query_parts = [query_understanding.intent, query_understanding.user_goal or "",
                       query_understanding.analysis_goal or ""]
        query = " ".join(part for part in query_parts if part).strip() or query_understanding.intent
        return VisRAGRequest(
            query=query,
            intent=query_understanding.intent,
            operations=list(query_understanding.requested_operations),
            candidate_charts=list(query_understanding.candidate_charts),
            selected_fields=list(request_analysis.selected_fields),
            data_profile=cls._to_core_data_profile(data_profile),
            top_k_examples=runtime.settings.visrag_top_k_examples,
            top_k_candidates=runtime.settings.visrag_top_k_recommendations,
        )

    @staticmethod
    def _to_core_data_profile(data_profile: DataProfile) -> VisRAGDataProfile:
        columns = [
            VisRAGColumnProfile(
                name=column.name,
                semantic_type=column.dtype,
                role=data_profile.field_roles.get(column.name),
                original_name=column.original_name,
                safe_name=column.safe_name,
                unique_count=column.unique_count,
                min_value=column.min_value,
                max_value=column.max_value,
                sample_values=list(column.sample_values),
                is_identifier=column.is_identifier,
                is_high_cardinality=column.is_high_cardinality,
            )
            for column in data_profile.columns
        ]
        return VisRAGDataProfile(
            columns=columns,
            row_count=data_profile.row_count,
            col_count=data_profile.col_count,
            field_roles=dict(data_profile.field_roles),
            numeric_columns=list(data_profile.likely_numeric_columns),
            categorical_columns=list(data_profile.likely_categorical_columns),
            temporal_columns=list(data_profile.likely_time_columns),
        )

    @classmethod
    def _to_domain_result(
            cls,
            core_result,
            query_understanding: QueryUnderstandingResult,
            request_analysis: RequestAnalysisResult,
            data_profile: DataProfile,
    ) -> VisRAGResult:
        recommendations = [
            VisRAGRecommendation(
                chart_family=candidate.recommended_mark,
                rationale=candidate.rationale,
                priority=index,
                score=candidate.score,
                support_examples=list(candidate.support_example_ids),
                instruction_highlights=cls._instruction_highlights(candidate),
            )
            for index, candidate in enumerate(core_result.candidates, start=1)
        ]
        plan = cls._build_visualization_plan(core_result.candidates, query_understanding, request_analysis,
                                             data_profile)
        candidate_spec_set = cls._build_candidate_spec_set(core_result.candidates, core_result.retrieved_examples, plan)
        caveats = cls._dedupe([
            *query_understanding.constraints,
            *query_understanding.ambiguity_notes,
            *request_analysis.ambiguity_report,
            *request_analysis.missing_fields,
            *core_result.caveats,
        ])
        return VisRAGResult(
            recommendations=recommendations,
            visualization_plan=plan,
            rules=[
                "Prefer specifications that map cleanly to Vega-Lite encodings and transforms.",
                "Use only fields present in the current data profile.",
                "Treat retrieved examples as guidance, not as final specifications.",
            ],
            caveats=caveats,
            implementation_notes=plan.build_instructions if plan else [],
            retrieved_examples=[cls._retrieved_example_to_domain(example) for example in
                                core_result.retrieved_examples],
            corpus_status=dict(core_result.corpus_status),
            retrieval_strategy=core_result.retrieval_strategy,
            retrieval_query=core_result.retrieval_query,
            candidate_spec_set=candidate_spec_set,
        )

    @classmethod
    def _build_visualization_plan(
            cls,
            candidates: list[CoreCandidate],
            query_understanding: QueryUnderstandingResult,
            request_analysis: RequestAnalysisResult,
            data_profile: DataProfile,
    ) -> VisualizationPlan | None:
        if not candidates:
            return None
        selected = candidates[0]
        bindings = cls._build_field_bindings(selected)
        transforms = cls._build_transforms(selected)
        axes = cls._build_axes(selected)
        title = cls._title_for_candidate(selected, request_analysis)
        build_instructions = [
            f"Use {selected.recommended_mark} mark.",
            "Use only fields from the current data profile.",
            "Keep data/datasets out of the generated spec; the pipeline attaches data.url later.",
        ]
        if selected.spec_template:
            build_instructions.append(
                "Use the retrieved spec template as a pattern, then adapt fields to the current dataset.")
        return VisualizationPlan(
            chart_family=selected.recommended_mark,
            visual_task=selected.chart_intent,
            goal=query_understanding.analysis_goal or query_understanding.user_goal or query_understanding.intent,
            title=title,
            description=selected.rationale,
            field_bindings=bindings,
            transforms=transforms,
            axes=axes,
            build_instructions=build_instructions,
            mark_hints=[selected.recommended_mark],
            renderer_hints=["Prefer readable axes and avoid label overlap."],
            evidence_example_ids=list(selected.support_example_ids),
            confidence=max(0.01, min(1.0, selected.score)),
            vega_lite_ready=True,
        )

    @classmethod
    def _build_candidate_spec_set(
            cls,
            candidates: list[CoreCandidate],
            retrieved_examples: list[VisRAGExample],
            plan: VisualizationPlan | None,
    ) -> CandidateSpecSet:
        specs: list[CandidateSpec] = []
        for index, candidate in enumerate(candidates, start=1):
            candidate_plan = plan.model_copy(deep=True) if plan is not None else None
            if candidate_plan is not None:
                candidate_plan.chart_family = candidate.recommended_mark
                candidate_plan.visual_task = candidate.chart_intent
                candidate_plan.mark_hints = [candidate.recommended_mark]
                candidate_plan.field_bindings = cls._build_field_bindings(candidate)
                candidate_plan.transforms = cls._build_transforms(candidate)
                candidate_plan.axes = cls._build_axes(candidate)
                candidate_plan.evidence_example_ids = list(candidate.support_example_ids)
                candidate_plan.confidence = max(0.01, min(1.0, candidate.score))
            specs.append(
                CandidateSpec(
                    spec_id=candidate.candidate_id or f"candidate-{index:02d}-{candidate.recommended_mark}",
                    chart_family=candidate.recommended_mark,
                    summary=candidate.rationale,
                    score=candidate.score,
                    rationale=candidate.rationale,
                    visualization_plan=candidate_plan,
                    spec_template=candidate.spec_template,
                    encoding_roles=dict(candidate.field_roles),
                    transform_types=list(candidate.transform_types),
                )
            )
        return CandidateSpecSet(
            candidate_specs=specs,
            retrieved_examples=[cls._retrieved_example_to_domain(example) for example in retrieved_examples],
            visualization_plan=plan,
            ranking_hints=[candidate.rationale for candidate in candidates],
            selected_candidate_spec=specs[0] if specs else None,
        )

    @staticmethod
    def _build_field_bindings(candidate: CoreCandidate) -> list[VisualizationFieldBinding]:
        bindings: list[VisualizationFieldBinding] = []
        for channel, field_name in candidate.field_mapping.items():
            field_role = candidate.field_roles.get(channel, "nominal")
            aggregate = None
            if channel == "y" and candidate.recommended_mark in {"bar", "line",
                                                                 "area"} and field_role == "quantitative":
                aggregate = "mean"
            bindings.append(
                VisualizationFieldBinding(
                    channel=channel,
                    field_name=field_name,
                    field_role=field_role,
                    title=field_name.replace("_", " "),
                    aggregate=aggregate,
                    required=channel in {"x", "y"},
                )
            )
        return bindings

    @staticmethod
    def _build_transforms(candidate: CoreCandidate) -> list[VisualizationTransform]:
        transforms: list[VisualizationTransform] = []
        for transform_type in candidate.transform_types:
            transforms.append(
                VisualizationTransform(kind=transform_type, description=f"Suggested by VisRAG: {transform_type}"))
        return transforms

    @staticmethod
    def _build_axes(candidate: CoreCandidate) -> list[VisualizationAxisInstruction]:
        axes: list[VisualizationAxisInstruction] = []
        for channel in ("x", "y"):
            field_name = candidate.field_mapping.get(channel)
            if not field_name:
                continue
            field_role = candidate.field_roles.get(channel, "nominal")
            axes.append(
                VisualizationAxisInstruction(
                    channel=channel,
                    field_name=field_name,
                    title=field_name.replace("_", " "),
                    scale_type=field_role,
                    format_hint="%Y-%m-%d" if field_role == "temporal" else None,
                    rotate_labels=(channel == "x" and field_role in {"nominal", "ordinal"}),
                )
            )
        return axes

    @staticmethod
    def _title_for_candidate(candidate: CoreCandidate, request_analysis: RequestAnalysisResult) -> str:
        fields = [field for field in (candidate.field_mapping.get("y"), candidate.field_mapping.get("x")) if field]
        if fields:
            return " by ".join(fields)
        if request_analysis.selected_fields:
            return " by ".join(request_analysis.selected_fields[:2])
        return f"{candidate.recommended_mark.title()} chart"

    @staticmethod
    def _instruction_highlights(candidate: CoreCandidate) -> list[str]:
        highlights = []
        if candidate.field_mapping:
            mapping = ", ".join(f"{channel}={field}" for channel, field in candidate.field_mapping.items())
            highlights.append(f"Suggested field mapping: {mapping}.")
        if candidate.spec_template:
            highlights.append("Retrieved candidate includes a Vega-Lite spec template.")
        return highlights

    @staticmethod
    def _retrieved_example_to_domain(example: VisRAGExample):
        from src.domain.models import VisRAGRetrievedExample

        return VisRAGRetrievedExample(
            example_id=example.example_id,
            source=example.source,
            corpus=example.corpus,
            chart_type=example.chart_type,
            instruction=example.instruction,
            description=example.description,
            tags=list(example.tags),
            code_language=example.code_language,
            domain=example.domain,
            score=example.score,
            rationale=example.rationale,
            document_id=example.document_id,
            metadata={
                **dict(example.metadata),
                "spec_template": example.spec_template,
                "field_roles": example.field_roles,
                "transform_types": example.transform_types,
            },
        )

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = value.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        return result
