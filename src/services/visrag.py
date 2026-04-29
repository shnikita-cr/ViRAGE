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

_SUPPORTED_PIPELINE_CHARTS = {"line", "bar", "scatter", "histogram", "boxplot", "area", "point", "circle", "tick"}
_FALLBACK_CHART_MAP = {"heatmap": "bar"}


class _PlanRefinementSchema(BaseModel):
    title: str = Field(min_length=1)
    subtitle: str | None = None
    description: str | None = None
    build_instructions: list[str] = Field(default_factory=list)
    mark_hints: list[str] = Field(default_factory=list)
    renderer_hints: list[str] = Field(default_factory=list)
    extra_caveats: list[str] = Field(default_factory=list)


class VisRAGService(BaseService):
    def invoke(self, query_understanding: QueryUnderstandingResult, request_analysis: RequestAnalysisResult,
               data_profile: DataProfile, runtime: RuntimeContext) -> VisRAGResult:
        candidate_charts, normalization_notes = self._normalize_candidate_charts(query_understanding.candidate_charts)
        if not candidate_charts:
            candidate_charts = ["bar", "line"]
        retrieval_queries = self._build_retrieval_queries(query_understanding, request_analysis, data_profile)
        retrieved = self._retrieve_multi_query(retrieval_queries, candidate_charts, runtime)
        support = summarize_chart_support(retrieved["examples"])
        recommendations = self._build_recommendations(query_understanding=query_understanding,
                                                      request_analysis=request_analysis, data_profile=data_profile,
                                                      candidate_charts=candidate_charts, support=support,
                                                      top_k=runtime.settings.visrag_top_k_recommendations)
        plan = self._build_visualization_plan(query_understanding=query_understanding,
                                              request_analysis=request_analysis, data_profile=data_profile,
                                              recommendations=recommendations, support=support)
        candidate_spec_set = self._build_candidate_spec_set(recommendations, retrieved["examples"], plan)
        caveats = self._dedupe(
            [*query_understanding.constraints, *query_understanding.ambiguity_notes, *request_analysis.ambiguity_report,
             *request_analysis.missing_fields, *normalization_notes])
        if not retrieved["examples"]:
            caveats.append("No local corpus matches were retrieved; VisRAG relied on data-aware ranking only.")
        if runtime.settings.visrag_embedding_backend == "local_tfidf":
            caveats.append("Retrieval is using the local TF-IDF backend.")
        if runtime.settings.visrag_enable_llm_synthesis:
            reasoning_llm = runtime.reasoning_llm
            if reasoning_llm is None:
                raise RuntimeError("VisRAG LLM synthesis is enabled, but runtime.reasoning_llm is missing.")
            plan, llm_caveats = self._refine_plan_with_llm(plan, query_understanding, retrieved["examples"],
                                                           reasoning_llm, runtime)
            caveats = self._dedupe([*caveats, *llm_caveats])
            candidate_spec_set.visualization_plan = plan
            if candidate_spec_set.selected_candidate_spec is not None:
                candidate_spec_set.selected_candidate_spec.visualization_plan = plan
        return VisRAGResult(recommendations=recommendations, visualization_plan=plan,
                            rules=["Prefer specifications that map cleanly to Vega-Lite encodings and transforms.",
                                   "Keep titles, axes and marks explicit so downstream generation stays deterministic.",
                                   "Prefer retrieved corpus examples when they agree with the detected data shape."],
                            caveats=self._dedupe(caveats), implementation_notes=plan.build_instructions,
                            retrieved_examples=[item.to_model() for item in retrieved["examples"]],
                            corpus_status=retrieved["corpus_status"], retrieval_strategy=(
                f"multi_query_{runtime.settings.visrag_embedding_backend}" if retrieved[
                    "examples"] else "heuristic_only"), retrieval_query=" || ".join(retrieval_queries),
                            candidate_spec_set=candidate_spec_set)

    def _build_retrieval_queries(self, query_understanding: QueryUnderstandingResult,
                                 request_analysis: RequestAnalysisResult, data_profile: DataProfile) -> list[str]:
        parts = [query_understanding.intent]
        if request_analysis.selected_fields:
            parts.append(f"selected fields: {', '.join(request_analysis.selected_fields)}")
        if data_profile.likely_time_columns:
            parts.append("time series temporal trend")
        if len(data_profile.likely_numeric_columns) >= 2:
            parts.append("numeric relationship")
        elif data_profile.likely_numeric_columns:
            parts.append("single numeric measure")
        queries = [" ".join(part for part in parts if part).strip()]
        for variant in query_understanding.query_variants:
            if variant.kind in {"canonical", "schema_grounding", "spec_retrieval", "analysis"}:
                queries.append(variant.text.strip())
        return self._dedupe([query for query in queries if query])

    def _retrieve_multi_query(self, queries: list[str], candidate_charts: list[str], runtime: RuntimeContext) -> dict[
        str, Any]:
        merged: dict[str, Any] = {}
        corpus_status: dict[str, str] = {}
        for query_text in queries:
            bundle = retrieve_examples(runtime.settings.visrag_corpus_root,
                                       index_root=runtime.settings.visrag_index_root, query_text=query_text,
                                       preferred_chart_types=candidate_charts,
                                       top_k=runtime.settings.visrag_top_k_examples,
                                       fetch_k=runtime.settings.visrag_retriever_fetch_k,
                                       similarity_threshold=runtime.settings.visrag_similarity_threshold,
                                       embedding_backend=runtime.settings.visrag_embedding_backend,
                                       embedding_model=runtime.settings.visrag_embedding_model,
                                       ollama_base_url=runtime.settings.visrag_ollama_base_url,
                                       ollama_timeout_seconds=runtime.settings.visrag_ollama_timeout_seconds,
                                       force_rebuild=runtime.settings.visrag_force_rebuild_index)
            corpus_status.update(bundle.corpus_status)
            for example in bundle.examples:
                current = merged.get(example.example_id)
                if current is None or current.score < example.score:
                    merged[example.example_id] = example
        examples = sorted(merged.values(), key=lambda item: (-item.score, item.chart_type, item.example_id))
        return {"examples": examples[: runtime.settings.visrag_top_k_examples], "corpus_status": corpus_status}

    def _build_recommendations(self, *, query_understanding: QueryUnderstandingResult,
                               request_analysis: RequestAnalysisResult, data_profile: DataProfile,
                               candidate_charts: list[str], support: dict[str, list[Any]], top_k: int) -> list[
        VisRAGRecommendation]:
        chart_pool = list(candidate_charts)
        for chart_type in support:
            if chart_type not in chart_pool and chart_type in _SUPPORTED_PIPELINE_CHARTS:
                chart_pool.append(chart_type)
        scored = []
        for index, chart_type in enumerate(chart_pool):
            heuristic_score, heuristic_reason = self._heuristic_score(chart_type=chart_type, index=index,
                                                                      query_understanding=query_understanding,
                                                                      request_analysis=request_analysis,
                                                                      data_profile=data_profile)
            evidence = support.get(chart_type, [])
            retrieval_score = self._retrieval_score(evidence)
            total_score = round(heuristic_score + retrieval_score, 4)
            instruction_highlights = self._instruction_highlights(chart_type, data_profile, request_analysis)
            rationale_parts = [heuristic_reason]
            if evidence:
                rationale_parts.append(f"{len(evidence)} retrieved corpus example(s) support this chart family.")
            scored.append((chart_type, total_score, " ".join(rationale_parts),
                           [item.example_id for item in evidence[:3]], instruction_highlights))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [VisRAGRecommendation(chart_family=chart_type, rationale=rationale, priority=priority, score=score,
                                     support_examples=support_examples, instruction_highlights=instruction_highlights)
                for priority, (chart_type, score, rationale, support_examples, instruction_highlights) in
                enumerate(scored[:top_k], start=1)]

    def _build_visualization_plan(self, *, query_understanding: QueryUnderstandingResult,
                                  request_analysis: RequestAnalysisResult, data_profile: DataProfile,
                                  recommendations: list[VisRAGRecommendation],
                                  support: dict[str, list[Any]]) -> VisualizationPlan:
        if not recommendations:
            raise RuntimeError("VisRAG could not produce any recommendations.")
        top = recommendations[0]
        chart_family = top.chart_family
        x_field, x_role, y_field, y_role, color_field, color_role = self._plan_channels(chart_family, data_profile,
                                                                                        request_analysis)
        bindings = [VisualizationFieldBinding(channel="x", field_name=x_field, field_role=x_role, title=x_field.title(),
                                              required=True),
                    VisualizationFieldBinding(channel="y", field_name=y_field, field_role=y_role, title=y_field.title(),
                                              aggregate="mean" if chart_family in {"line", "bar", "area"} else None,
                                              required=True)]
        axes = [VisualizationAxisInstruction(channel="x", field_name=x_field, title=x_field.title(), scale_type=(
            "temporal" if x_role == "temporal" else "linear" if x_role == "quantitative" else "categorical"),
                                             rotate_labels=x_role == "temporal"),
                VisualizationAxisInstruction(channel="y", field_name=y_field, title=y_field.title(),
                                             scale_type="linear")]
        if color_field:
            bindings.append(VisualizationFieldBinding(channel="color", field_name=color_field, field_role=color_role,
                                                      title=color_field.title(), required=False))
        transforms = []
        if chart_family == "histogram":
            transforms.append(VisualizationTransform(kind="bin", field_name=x_field,
                                                     description=f"Bin {x_field} into histogram buckets."))
        build_instructions = [*top.instruction_highlights,
                              "Use explicit encodings so the final Vega-Lite spec remains deterministic."]
        evidence_ids = [item.example_id for item in support.get(chart_family, [])[:3]]
        return VisualizationPlan(chart_family=chart_family,
                                 visual_task=query_understanding.task_type or query_understanding.intent,
                                 goal=query_understanding.user_goal or query_understanding.intent,
                                 title=query_understanding.intent, description=top.rationale, field_bindings=bindings,
                                 transforms=transforms, axes=axes, build_instructions=self._dedupe(build_instructions),
                                 mark_hints=[chart_family], renderer_hints=["Prefer readable defaults."],
                                 evidence_example_ids=evidence_ids, confidence=max(0.01, min(1.0, top.score)),
                                 vega_lite_ready=True)

    def _build_candidate_spec_set(self, recommendations: list[VisRAGRecommendation], retrieved_examples: list[Any],
                                  plan: VisualizationPlan) -> CandidateSpecSet:
        specs = []
        for index, item in enumerate(recommendations, start=1):
            candidate_plan = plan.model_copy(deep=True)
            candidate_plan.chart_family = item.chart_family
            candidate_plan.mark_hints = [item.chart_family]
            specs.append(
                CandidateSpec(spec_id=f"candidate-{index:02d}-{item.chart_family}", chart_family=item.chart_family,
                              summary=item.rationale, score=item.score, rationale=item.rationale,
                              visualization_plan=candidate_plan))
        return CandidateSpecSet(candidate_specs=specs,
                                retrieved_examples=[example.to_model() for example in retrieved_examples],
                                visualization_plan=plan, ranking_hints=[item.rationale for item in recommendations],
                                selected_candidate_spec=(specs[0] if specs else None))

    def _plan_channels(self, chart_family: str, data_profile: DataProfile, request_analysis: RequestAnalysisResult) -> \
            tuple[str, str, str, str, str | None, str]:
        selected = list(request_analysis.selected_fields)
        time_columns = data_profile.likely_time_columns
        numeric_columns = data_profile.likely_numeric_columns
        categorical_columns = data_profile.likely_categorical_columns
        if chart_family in {"line", "area"} and time_columns and numeric_columns:
            color_field = next((field for field in selected if field in categorical_columns),
                               categorical_columns[0] if categorical_columns else None)
            return time_columns[0], "temporal", numeric_columns[0], "quantitative", color_field, "nominal"
        if chart_family == "scatter" and len(numeric_columns) >= 2:
            return numeric_columns[0], "quantitative", numeric_columns[1], "quantitative", None, "nominal"
        if chart_family in {"bar", "boxplot", "tick"} and categorical_columns and numeric_columns:
            return categorical_columns[0], "nominal", numeric_columns[0], "quantitative", None, "nominal"
        if chart_family == "histogram" and numeric_columns:
            return numeric_columns[0], "quantitative", numeric_columns[0], "quantitative", None, "nominal"
        if numeric_columns:
            x_fallback = numeric_columns[0]
            y_fallback = numeric_columns[1] if len(numeric_columns) >= 2 else numeric_columns[0]
            return x_fallback, "quantitative", y_fallback, "quantitative", None, "nominal"
        first = selected[0] if selected else "value"
        return first, "nominal", first, "quantitative", None, "nominal"

    def _heuristic_score(self, *, chart_type: str, index: int, query_understanding: QueryUnderstandingResult,
                         request_analysis: RequestAnalysisResult, data_profile: DataProfile) -> tuple[float, str]:
        query_text = " ".join([query_understanding.intent, *query_understanding.requested_operations,
                               *request_analysis.grounded_fields]).lower()
        base = max(0.05, 0.2 - index * 0.02)
        if chart_type in {"line", "area"} and data_profile.likely_time_columns and data_profile.likely_numeric_columns:
            return base + 0.75, "Time-like field and numeric measure detected; trend chart is a strong fit."
        if chart_type == "scatter" and len(data_profile.likely_numeric_columns) >= 2:
            return base + 0.70, "Two numeric fields detected; scatter fits relationship analysis."
        if chart_type in {"bar", "boxplot",
                          "tick"} and data_profile.likely_categorical_columns and data_profile.likely_numeric_columns:
            return base + 0.62, "Categorical and numeric fields detected; comparison chart is a good fit."
        if chart_type == "histogram" and data_profile.likely_numeric_columns:
            return base + 0.55, "Numeric measure detected; histogram fits distribution analysis."
        if chart_type in query_text:
            return base + 0.30, f"User language explicitly mentions or strongly implies '{chart_type}'."
        return base, f"{chart_type} remains a fallback recommendation based on the detected task."

    @staticmethod
    def _retrieval_score(evidence: list[Any]) -> float:
        if not evidence:
            return 0.0
        strongest = evidence[0].score
        diversity_bonus = min(0.20, 0.03 * len(evidence))
        return strongest + diversity_bonus

    def _instruction_highlights(self, chart_type: str, data_profile: DataProfile,
                                request_analysis: RequestAnalysisResult) -> list[str]:
        if chart_type in {"line", "area"}:
            x = data_profile.likely_time_columns[0] if data_profile.likely_time_columns else (
                request_analysis.selected_fields[0] if request_analysis.selected_fields else "date")
            y = data_profile.likely_numeric_columns[0] if data_profile.likely_numeric_columns else "value"
            hints = [f"Place {x} on the x-axis and {y} on the y-axis."]
            if data_profile.likely_categorical_columns:
                hints.append(f"Use {data_profile.likely_categorical_columns[0]} as an optional color grouping.")
            return hints
        if chart_type == "scatter" and len(data_profile.likely_numeric_columns) >= 2:
            x, y = data_profile.likely_numeric_columns[:2]
            return [f"Plot {x} against {y} using point marks."]
        if chart_type == "histogram":
            x = data_profile.likely_numeric_columns[0] if data_profile.likely_numeric_columns else "value"
            return [f"Bin the quantitative field {x} along the x-axis."]
        x = data_profile.likely_categorical_columns[0] if data_profile.likely_categorical_columns else "category"
        y = data_profile.likely_numeric_columns[0] if data_profile.likely_numeric_columns else "value"
        return [f"Aggregate {y} by {x}."]

    def _refine_plan_with_llm(self, plan: VisualizationPlan, query_understanding: QueryUnderstandingResult,
                              retrieved_examples: list[Any], reasoning_llm: Any, runtime: RuntimeContext) -> tuple[
        VisualizationPlan, list[str]]:
        example_text = "\n".join(
            f"- {item.example_id} | {item.chart_type} | {item.instruction} | score={item.score:.3f}" for item in
            retrieved_examples[:3]) or "- none"
        prompt = (
            "You refine a visualization plan for a deterministic Vega-Lite generation pipeline.\n"
            "Do not change the chosen chart family or field bindings.\n"
            "Return only structured output with better title, subtitle, build instructions, mark hints and renderer hints.\n"
            f"User intent: {query_understanding.intent}\n"
            f"Base plan JSON: {plan.model_dump_json(indent=2)}\n"
            f"Retrieved evidence:\n{example_text}\n"
        )
        parsed = invoke_structured(reasoning_llm, prompt, _PlanRefinementSchema, runtime=runtime,
                                   stage="visrag_refinement", role="reasoning", examples=[
                {"title": plan.title, "subtitle": plan.subtitle or "", "description": plan.description or plan.goal,
                 "build_instructions": plan.build_instructions[:2] or ["Keep encodings explicit."],
                 "mark_hints": plan.mark_hints[:2] or [plan.chart_family],
                 "renderer_hints": plan.renderer_hints[:2] or ["Prefer readable defaults."], "extra_caveats": []}],
                                   max_attempts=2)
        plan.title = parsed.title or plan.title
        if parsed.subtitle:
            plan.subtitle = parsed.subtitle
        if parsed.description:
            plan.description = parsed.description
        plan.build_instructions = self._dedupe([*plan.build_instructions, *parsed.build_instructions])
        plan.mark_hints = self._dedupe([*plan.mark_hints, *parsed.mark_hints])
        plan.renderer_hints = self._dedupe([*plan.renderer_hints, *parsed.renderer_hints])
        return plan, self._dedupe(parsed.extra_caveats)

    @staticmethod
    def _normalize_candidate_charts(candidate_charts: list[str]) -> tuple[list[str], list[str]]:
        normalized = []
        notes = []
        seen: set[str] = set()
        for raw_chart in candidate_charts:
            raw = raw_chart.strip().lower()
            chart = canonicalize_chart_type(raw) or _FALLBACK_CHART_MAP.get(raw,
                                                                            raw if raw in _SUPPORTED_PIPELINE_CHARTS else "")
            if raw in _FALLBACK_CHART_MAP:
                notes.append(f"Chart family '{raw}' is not directly supported; using '{chart}' fallback.")
            if chart and chart not in seen:
                seen.add(chart)
                normalized.append(chart)
        return normalized, notes

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result = []
        for value in values:
            normalized = value.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                result.append(normalized)
        return result
