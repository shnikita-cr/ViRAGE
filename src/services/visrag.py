from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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
from src.llm.helpers import invoke_structured
from src.services.base import BaseService
from src.services.visrag_retrieval import canonicalize_chart_type, retrieve_examples, summarize_chart_support

_SUPPORTED_PIPELINE_CHARTS = {"line", "bar", "scatter", "histogram", "boxplot"}
_FALLBACK_CHART_MAP = {"area": "line", "heatmap": "bar"}


class _PlanRefinementSchema(BaseModel):
    title: str = Field(min_length=1)
    subtitle: str | None = None
    description: str | None = None
    build_instructions: list[str] = Field(default_factory=list)
    mark_hints: list[str] = Field(default_factory=list)
    renderer_hints: list[str] = Field(default_factory=list)
    extra_caveats: list[str] = Field(default_factory=list)


class VisRAGService(BaseService):
    def invoke(
        self,
        query_understanding: QueryUnderstandingResult,
        request_analysis: RequestAnalysisResult,
        data_profile: DataProfile,
        runtime: RuntimeContext,
    ) -> VisRAGResult:
        candidate_charts, normalization_notes = self._normalize_candidate_charts(query_understanding.candidate_charts)
        if not candidate_charts:
            candidate_charts = ["bar", "line"]

        retrieval_query = self._build_retrieval_query(query_understanding, request_analysis, data_profile)
        retrieved = retrieve_examples(
            runtime.settings.visrag_corpus_root,
            index_root=runtime.settings.visrag_index_root,
            query_text=retrieval_query,
            preferred_chart_types=candidate_charts,
            top_k=runtime.settings.visrag_top_k_examples,
            fetch_k=runtime.settings.visrag_retriever_fetch_k,
            similarity_threshold=runtime.settings.visrag_similarity_threshold,
            embedding_backend=runtime.settings.visrag_embedding_backend,
            embedding_model=runtime.settings.visrag_embedding_model,
            ollama_base_url=runtime.settings.visrag_ollama_base_url,
            ollama_timeout_seconds=runtime.settings.visrag_ollama_timeout_seconds,
            force_rebuild=runtime.settings.visrag_force_rebuild_index,
        )
        support = summarize_chart_support(retrieved.examples)
        recommendations = self._build_recommendations(
            query_understanding=query_understanding,
            request_analysis=request_analysis,
            data_profile=data_profile,
            candidate_charts=candidate_charts,
            support=support,
            top_k=runtime.settings.visrag_top_k_recommendations,
        )
        plan = self._build_visualization_plan(
            query_understanding=query_understanding,
            request_analysis=request_analysis,
            data_profile=data_profile,
            recommendations=recommendations,
            support=support,
        )
        candidate_spec_set = self._build_candidate_spec_set(recommendations, retrieved.examples, plan)

        caveats = self._dedupe([
            *query_understanding.constraints,
            *request_analysis.ambiguity_report,
            *request_analysis.missing_fields,
            *normalization_notes,
        ])
        if not retrieved.examples:
            caveats.append("No local corpus matches were retrieved; VisRAG relied on data-aware ranking only.")
        if retrieved.backend_name == "local_tfidf":
            caveats.append("Semantic retrieval is running on the local TF-IDF backend; enable dense embeddings for stronger recall.")

        if runtime.settings.visrag_enable_llm_synthesis:
            reasoning_llm = runtime.reasoning_llm
            if reasoning_llm is None:
                raise RuntimeError("VisRAG LLM synthesis is enabled, but runtime.reasoning_llm is missing.")
            plan, llm_caveats = self._refine_plan_with_llm(plan, query_understanding, retrieved.examples, reasoning_llm)
            caveats.extend(llm_caveats)
            candidate_spec_set.visualization_plan = plan
            if candidate_spec_set.selected_candidate_spec is not None:
                candidate_spec_set.selected_candidate_spec.visualization_plan = plan

        return VisRAGResult(
            recommendations=recommendations,
            visualization_plan=plan,
            rules=[
                "Prefer specifications that map cleanly to Vega-Lite encodings and transforms.",
                "Keep titles, axes and marks explicit so downstream generation stays deterministic.",
                "Prefer retrieved corpus examples when they agree with the detected data shape.",
            ],
            caveats=self._dedupe(caveats),
            implementation_notes=plan.build_instructions,
            retrieved_examples=[item.to_model() for item in retrieved.examples],
            corpus_status=retrieved.corpus_status,
            retrieval_strategy=f"semantic_{retrieved.backend_name}_plus_heuristics" if retrieved.examples else "heuristic_only",
            retrieval_query=retrieval_query,
            candidate_spec_set=candidate_spec_set,
        )

    def _build_recommendations(
        self,
        *,
        query_understanding: QueryUnderstandingResult,
        request_analysis: RequestAnalysisResult,
        data_profile: DataProfile,
        candidate_charts: list[str],
        support: dict[str, list[Any]],
        top_k: int,
    ) -> list[VisRAGRecommendation]:
        chart_pool = list(candidate_charts)
        for chart_type in support.keys():
            if chart_type not in chart_pool and chart_type in _SUPPORTED_PIPELINE_CHARTS:
                chart_pool.append(chart_type)

        scored: list[tuple[str, float, str, list[str], list[str]]] = []
        for index, chart_type in enumerate(chart_pool):
            heuristic_score, heuristic_reason = self._heuristic_score(
                chart_type=chart_type,
                index=index,
                query_understanding=query_understanding,
                request_analysis=request_analysis,
                data_profile=data_profile,
            )
            evidence = support.get(chart_type, [])
            retrieval_score = self._retrieval_score(evidence)
            total_score = round(heuristic_score + retrieval_score, 4)
            instruction_highlights = self._instruction_highlights(chart_type, data_profile, request_analysis)
            rationale_parts = [heuristic_reason]
            if evidence:
                rationale_parts.append(f"{len(evidence)} retrieved corpus example(s) support this chart family.")
            scored.append((chart_type, total_score, " ".join(rationale_parts), [item.example_id for item in evidence[:3]], instruction_highlights))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            VisRAGRecommendation(
                chart_family=chart_type,
                rationale=rationale,
                priority=priority,
                score=score,
                support_examples=support_examples,
                instruction_highlights=instruction_highlights,
            )
            for priority, (chart_type, score, rationale, support_examples, instruction_highlights) in enumerate(scored[:top_k], start=1)
        ]

    def _build_visualization_plan(
        self,
        *,
        query_understanding: QueryUnderstandingResult,
        request_analysis: RequestAnalysisResult,
        data_profile: DataProfile,
        recommendations: list[VisRAGRecommendation],
        support: dict[str, list[Any]],
    ) -> VisualizationPlan:
        chart_family = recommendations[0].chart_family if recommendations else "bar"
        visual_task = query_understanding.requested_operations[0] if query_understanding.requested_operations else "exploratory analysis"
        title = query_understanding.intent.strip() or "Generated visualization"
        subtitle = f"Task: {query_understanding.task_type or 'visual analysis'}; preferred chart family: {chart_family}."
        field_bindings, transforms, axes, build_instructions, mark_hints, renderer_hints = self._plan_channels(chart_family, data_profile, request_analysis)
        evidence_ids = [item.example_id for item in support.get(chart_family, [])[:3]]
        description = (
            f"Build a {chart_family} chart for {visual_task}. "
            f"Preserve explicit encodings so the plan can be converted to code or Vega-Lite later."
        )
        confidence = round(min(0.99, recommendations[0].score / 2.0 if recommendations else 0.45), 3)
        return VisualizationPlan(
            chart_family=chart_family,
            visual_task=visual_task,
            goal=query_understanding.intent,
            title=title,
            subtitle=subtitle,
            description=description,
            field_bindings=field_bindings,
            transforms=transforms,
            axes=axes,
            filters=list(query_understanding.constraints),
            build_instructions=self._dedupe(build_instructions),
            mark_hints=self._dedupe(mark_hints),
            renderer_hints=self._dedupe(renderer_hints),
            evidence_example_ids=evidence_ids,
            confidence=confidence,
            vega_lite_ready=True,
        )

    def _build_candidate_spec_set(
        self,
        recommendations: list[VisRAGRecommendation],
        retrieved_examples: list[Any],
        plan: VisualizationPlan,
    ) -> CandidateSpecSet:
        candidate_specs = [
            CandidateSpec(
                spec_id=f"candidate_{rec.priority}_{rec.chart_family}",
                chart_family=rec.chart_family,
                summary=rec.rationale,
                score=rec.score,
                rationale=rec.rationale,
                visualization_plan=plan if rec.priority == 1 else None,
            )
            for rec in recommendations
        ]
        selected = candidate_specs[0] if candidate_specs else None
        return CandidateSpecSet(
            candidate_specs=candidate_specs,
            retrieved_examples=[item.to_model() for item in retrieved_examples],
            visualization_plan=plan,
            ranking_hints=[rec.rationale for rec in recommendations[:2]],
            selected_candidate_spec=selected,
        )

    def _plan_channels(
        self,
        chart_family: str,
        data_profile: DataProfile,
        request_analysis: RequestAnalysisResult,
    ) -> tuple[list[VisualizationFieldBinding], list[VisualizationTransform], list[VisualizationAxisInstruction], list[str], list[str], list[str]]:
        numeric = [col for col in request_analysis.selected_fields if col in data_profile.likely_numeric_columns] or list(data_profile.likely_numeric_columns)
        categorical = [col for col in request_analysis.selected_fields if col in data_profile.likely_categorical_columns] or list(data_profile.likely_categorical_columns)
        temporal = [col for col in request_analysis.selected_fields if col in data_profile.likely_time_columns] or list(data_profile.likely_time_columns)
        encodings: list[VisualizationFieldBinding] = []
        transforms: list[VisualizationTransform] = []
        axes: list[VisualizationAxisInstruction] = []
        instructions: list[str] = []
        mark_hints: list[str] = []
        renderer_hints: list[str] = []

        def bind(channel: str, field_name: str, field_role: str, *, aggregate: str | None = None, title: str | None = None, time_unit: str | None = None) -> None:
            encodings.append(VisualizationFieldBinding(channel=channel, field_name=field_name, field_role=field_role, title=title or field_name, aggregate=aggregate, time_unit=time_unit))

        if chart_family == "line":
            x_field = temporal[0] if temporal else (categorical[0] if categorical else "index")
            y_field = numeric[0] if numeric else "value"
            bind("x", x_field, "temporal" if temporal else "nominal")
            bind("y", y_field, "quantitative")
            axes.extend([
                VisualizationAxisInstruction(channel="x", field_name=x_field, title=x_field, scale_type="temporal" if temporal else "categorical", rotate_labels=bool(temporal or categorical)),
                VisualizationAxisInstruction(channel="y", field_name=y_field, title=y_field, scale_type="linear"),
            ])
            transforms.append(VisualizationTransform(kind="aggregate", field_name=y_field, aggregate="mean", group_by=[x_field], description="Aggregate the numeric measure by the x axis before plotting when duplicate x values appear."))
            instructions.extend([
                f"Bind the x-axis to '{x_field}'.",
                f"Bind the y-axis to '{y_field}' as the primary quantitative measure.",
                f"Aggregate '{y_field}' by '{x_field}' with mean when duplicate x values appear.",
            ])
            mark_hints.extend(["Use a line mark with point markers.", "Preserve chronological order on the x-axis."])
            renderer_hints.extend(["Rotate x-axis labels if dates are dense.", "Tighten layout after formatting dates."])
        elif chart_family == "scatter":
            x_field = numeric[0] if numeric else "x_value"
            y_field = numeric[1] if len(numeric) >= 2 else (numeric[0] if numeric else "y_value")
            bind("x", x_field, "quantitative")
            bind("y", y_field, "quantitative")
            if categorical:
                bind("color", categorical[0], "nominal")
            axes.extend([
                VisualizationAxisInstruction(channel="x", field_name=x_field, title=x_field, scale_type="linear"),
                VisualizationAxisInstruction(channel="y", field_name=y_field, title=y_field, scale_type="linear"),
            ])
            instructions.extend([f"Use '{x_field}' on the x-axis and '{y_field}' on the y-axis.", "Keep raw points without aggregation unless later requirements demand summarization."])
            if categorical:
                instructions.append(f"Use '{categorical[0]}' as the grouping or color channel when plotting points.")
            mark_hints.extend(["Use a scatter mark.", "Consider semi-transparent points if the plot is dense."])
        elif chart_family == "histogram":
            x_field = numeric[0] if numeric else "value"
            bind("x", x_field, "quantitative")
            bind("y", "count", "quantitative", aggregate="count", title="count")
            axes.extend([
                VisualizationAxisInstruction(channel="x", field_name=x_field, title=x_field, scale_type="linear"),
                VisualizationAxisInstruction(channel="y", field_name="count", title="count", scale_type="linear"),
            ])
            transforms.append(VisualizationTransform(kind="bin", field_name=x_field, description="Bin the numeric values before counting observations."))
            transforms.append(VisualizationTransform(kind="aggregate", field_name="count", aggregate="count", group_by=[x_field], description="Count observations per histogram bin."))
            instructions.extend([f"Use '{x_field}' as the quantitative field for the histogram bins.", "Count observations per bin on the y-axis."])
            mark_hints.append("Use a histogram mark or bar mark over binned values.")
        elif chart_family == "boxplot":
            x_field = categorical[0] if categorical else "group"
            y_field = numeric[0] if numeric else "value"
            bind("x", x_field, "nominal")
            bind("y", y_field, "quantitative")
            axes.extend([
                VisualizationAxisInstruction(channel="x", field_name=x_field, title=x_field, scale_type="categorical", rotate_labels=True),
                VisualizationAxisInstruction(channel="y", field_name=y_field, title=y_field, scale_type="linear"),
            ])
            instructions.extend([f"Group the distribution by '{x_field}'.", f"Use '{y_field}' as the quantitative spread shown by the boxplot."])
            mark_hints.append("Use a boxplot mark to emphasize median, quartiles and outliers.")
        else:
            x_field = categorical[0] if categorical else (temporal[0] if temporal else "category")
            y_field = numeric[0] if numeric else "value"
            bind("x", x_field, "nominal" if categorical else "temporal")
            bind("y", y_field, "quantitative", aggregate="mean")
            axes.extend([
                VisualizationAxisInstruction(channel="x", field_name=x_field, title=x_field, scale_type="categorical" if categorical else "temporal", rotate_labels=True),
                VisualizationAxisInstruction(channel="y", field_name=y_field, title=y_field, scale_type="linear"),
            ])
            transforms.append(VisualizationTransform(kind="aggregate", field_name=y_field, aggregate="mean", group_by=[x_field], description="Aggregate the measure by category before plotting bars."))
            instructions.extend([f"Use '{x_field}' as the discrete comparison axis.", f"Use '{y_field}' as the quantitative bar height after aggregation."])
            mark_hints.append("Use a bar mark with one bar per grouped category.")
            renderer_hints.append("Sort categories if ranking clarity matters.")
        return encodings, transforms, axes, instructions, mark_hints, renderer_hints

    def _heuristic_score(
        self,
        *,
        chart_type: str,
        index: int,
        query_understanding: QueryUnderstandingResult,
        request_analysis: RequestAnalysisResult,
        data_profile: DataProfile,
    ) -> tuple[float, str]:
        base = max(0.08, 0.40 - index * 0.05)
        ops_text = " ".join(query_understanding.requested_operations).lower()
        if request_analysis.ambiguity_report:
            base -= 0.03
        if chart_type == "line":
            if data_profile.likely_time_columns and data_profile.likely_numeric_columns:
                return base + 0.60, "Time-like and numeric fields detected; line best matches a trend-oriented request."
            return base + 0.15, "Line retained as a trend-oriented option."
        if chart_type == "scatter":
            if len(data_profile.likely_numeric_columns) >= 2:
                return base + 0.55, "Two numeric fields detected; scatter suits relationship analysis."
            return base + 0.10, "Scatter retained as a relationship-focused option."
        if chart_type == "histogram":
            if data_profile.likely_numeric_columns:
                return base + 0.45, "Numeric field detected; histogram suits distribution analysis."
            return base + 0.05, "Histogram retained with weak numeric support."
        if chart_type == "boxplot":
            if data_profile.likely_categorical_columns and data_profile.likely_numeric_columns:
                return base + 0.45, "Categorical and numeric fields detected; boxplot suits grouped spread analysis."
            return base + 0.08, "Boxplot retained with weak grouping support."
        if data_profile.likely_categorical_columns and data_profile.likely_numeric_columns:
            return base + 0.45, "Categorical and numeric fields detected; bar chart suits grouped comparison."
        if "comparison" in ops_text or "ranking" in ops_text:
            return base + 0.25, "Bar chart suits comparison or ranking requests."
        return base + 0.15, "Bar chart retained as a conservative default."

    @staticmethod
    def _retrieval_score(evidence: list[Any]) -> float:
        if not evidence:
            return 0.0
        top_scores = [item.score for item in evidence[:3]]
        return round(sum(top_scores) / len(top_scores), 4)

    @staticmethod
    def _build_retrieval_query(
        query_understanding: QueryUnderstandingResult,
        request_analysis: RequestAnalysisResult,
        data_profile: DataProfile,
    ) -> str:
        parts: list[str] = [
            f"intent: {query_understanding.intent}",
            f"operations: {', '.join(query_understanding.requested_operations)}",
            f"candidate_charts: {', '.join(query_understanding.candidate_charts)}",
            f"grounded_fields: {', '.join(request_analysis.grounded_fields)}",
            f"selected_fields: {', '.join(request_analysis.selected_fields)}",
        ]
        if data_profile.likely_time_columns:
            parts.append("data_shape: temporal x-axis available")
        if len(data_profile.likely_numeric_columns) >= 2:
            parts.append("data_shape: two quantitative fields available")
        elif data_profile.likely_numeric_columns:
            parts.append("data_shape: one quantitative measure available")
        if data_profile.likely_categorical_columns:
            parts.append("data_shape: categorical grouping available")
        return " | ".join(part for part in parts if part)

    @staticmethod
    def _normalize_candidate_charts(candidate_charts: list[str]) -> tuple[list[str], list[str]]:
        normalized: list[str] = []
        notes: list[str] = []
        seen: set[str] = set()
        for raw_chart in candidate_charts:
            raw = raw_chart.strip().lower()
            chart = canonicalize_chart_type(raw)
            if not chart and raw in _FALLBACK_CHART_MAP:
                chart = _FALLBACK_CHART_MAP[raw]
                notes.append(f"Chart family '{raw}' is not directly supported by the current renderer; using '{chart}' instead.")
            elif not chart:
                chart = raw if raw in _SUPPORTED_PIPELINE_CHARTS else ""
            if chart and chart not in seen:
                seen.add(chart)
                normalized.append(chart)
        return normalized, notes

    @staticmethod
    def _instruction_highlights(chart_type: str, data_profile: DataProfile, request_analysis: RequestAnalysisResult) -> list[str]:
        selected = request_analysis.selected_fields
        if chart_type == "line":
            x = next((field for field in selected if field in data_profile.likely_time_columns), None) or (data_profile.likely_time_columns[0] if data_profile.likely_time_columns else "index")
            y = next((field for field in selected if field in data_profile.likely_numeric_columns), None) or (data_profile.likely_numeric_columns[0] if data_profile.likely_numeric_columns else "value")
            return [f"Use {x} on the x-axis.", f"Use {y} on the y-axis."]
        if chart_type == "scatter" and len(data_profile.likely_numeric_columns) >= 2:
            return [f"Use {data_profile.likely_numeric_columns[0]} vs {data_profile.likely_numeric_columns[1]}."]
        if chart_type == "histogram" and data_profile.likely_numeric_columns:
            return [f"Bin {data_profile.likely_numeric_columns[0]} and count observations."]
        if chart_type == "boxplot" and data_profile.likely_numeric_columns:
            group = data_profile.likely_categorical_columns[0] if data_profile.likely_categorical_columns else "group"
            return [f"Show the spread of {data_profile.likely_numeric_columns[0]} grouped by {group}."]
        x = data_profile.likely_categorical_columns[0] if data_profile.likely_categorical_columns else "category"
        y = data_profile.likely_numeric_columns[0] if data_profile.likely_numeric_columns else "value"
        return [f"Aggregate {y} by {x}."]

    def _refine_plan_with_llm(
        self,
        plan: VisualizationPlan,
        query_understanding: QueryUnderstandingResult,
        retrieved_examples: list[Any],
        reasoning_llm: Any,
    ) -> tuple[VisualizationPlan, list[str]]:
        example_text = "\n".join(
            f"- {item.example_id} | {item.chart_type} | {item.instruction} | score={item.score:.3f}" for item in retrieved_examples[:3]
        ) or "- none"
        prompt = (
            "You refine a visualization plan for a code generation pipeline.\n"
            "Do not change the chosen chart family or field bindings.\n"
            "Return only structured output with better title, subtitle, build instructions, mark hints and renderer hints.\n"
            f"User intent: {query_understanding.intent}\n"
            f"Base plan JSON: {plan.model_dump_json(indent=2)}\n"
            f"Retrieved evidence:\n{example_text}\n"
        )
        parsed = invoke_structured(reasoning_llm, prompt, _PlanRefinementSchema)
        plan.title = parsed.title or plan.title
        if parsed.subtitle:
            plan.subtitle = parsed.subtitle
        if parsed.description:
            plan.description = parsed.description
        plan.build_instructions = self._dedupe([*plan.build_instructions, *parsed.build_instructions])
        plan.mark_hints = self._dedupe([*plan.mark_hints, *parsed.mark_hints])
        plan.renderer_hints = self._dedupe([*plan.renderer_hints, *parsed.renderer_hints])
        return plan, parsed.extra_caveats

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
