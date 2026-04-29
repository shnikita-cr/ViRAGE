from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.visrag_core.corpus import canonicalize_chart_type
from src.visrag_core.models import VisRAGCandidate, VisRAGConfig, VisRAGDataProfile, VisRAGRequest, VisRAGResult
from src.visrag_core.retriever import retrieve_examples, summarize_chart_support

_SUPPORTED_CHARTS = {"line", "area", "bar", "point", "histogram", "boxplot", "circle", "tick"}
_DEFAULT_CHARTS = ["bar", "line", "point", "histogram"]


class VisRAGCoreService:
    """Portable RAG service for chart-pattern retrieval.

    This service has no dependency on ViRAGE runtime, artifacts, LangChain,
    application state or domain models. It receives portable DTOs and returns
    portable DTOs.
    """

    def __init__(self, config: VisRAGConfig | None = None) -> None:
        self.config = config or VisRAGConfig()

    def search(self, request: VisRAGRequest) -> VisRAGResult:
        charts, normalization_notes = self._normalize_chart_candidates(request.candidate_charts)
        if not charts:
            charts = list(_DEFAULT_CHARTS)

        queries = self._build_queries(request)
        retrieved_by_id = {}
        corpus_status: dict[str, str] = {}
        backend_name = "none"
        for query in queries:
            bundle = retrieve_examples(
                self.config.corpus_root,
                index_root=self.config.index_root,
                query_text=query,
                preferred_chart_types=charts,
                top_k=request.top_k_examples,
                fetch_k=self.config.retriever_fetch_k,
                similarity_threshold=self.config.similarity_threshold,
                embedding_backend=self.config.embedding_backend,
                embedding_model=self.config.embedding_model,
                ollama_base_url=self.config.ollama_base_url,
                ollama_timeout_seconds=self.config.ollama_timeout_seconds,
                force_rebuild=self.config.force_rebuild_index,
            )
            backend_name = bundle.backend_name
            corpus_status.update(bundle.corpus_status)
            for example in bundle.examples:
                current = retrieved_by_id.get(example.example_id)
                if current is None or current.score < example.score:
                    retrieved_by_id[example.example_id] = example

        retrieved = sorted(retrieved_by_id.values(), key=lambda item: (-item.score, item.chart_type, item.example_id))[
                    : request.top_k_examples]
        support = summarize_chart_support(retrieved)
        candidates = self._build_candidates(request=request, chart_pool=charts, support=support)
        caveats = list(normalization_notes)
        if not retrieved:
            caveats.append("No corpus examples were retrieved; candidates were built with data-aware heuristics.")
        if self.config.embedding_backend == "local_tfidf":
            caveats.append("Retrieval is using the local TF-IDF backend.")
        return VisRAGResult(
            candidates=candidates[: request.top_k_candidates],
            retrieved_examples=retrieved,
            corpus_status=corpus_status,
            retrieval_strategy=f"multi_query_{backend_name}" if retrieved else "heuristic_only",
            retrieval_query=" || ".join(queries),
            caveats=self._dedupe(caveats),
        )

    def _build_queries(self, request: VisRAGRequest) -> list[str]:
        profile = request.data_profile
        parts = [request.query, request.intent, " ".join(request.operations)]
        if request.selected_fields:
            parts.append(f"selected fields: {', '.join(request.selected_fields)}")
        if profile.temporal_columns:
            parts.append("temporal trend time series")
        if len(profile.numeric_columns) >= 2:
            parts.append("numeric relationship correlation scatter")
        elif profile.numeric_columns:
            parts.append("single numeric measure aggregate")
        if profile.categorical_columns:
            parts.append("category comparison grouping")
        primary = " ".join(part for part in parts if part).strip()
        variants = [primary]
        if request.intent and request.intent != request.query:
            variants.append(request.intent)
        return self._dedupe([item for item in variants if item]) or [request.query]

    def _build_candidates(self, *, request: VisRAGRequest, chart_pool: list[str], support: dict[str, list[Any]]) -> \
    list[VisRAGCandidate]:
        ranked: list[VisRAGCandidate] = []
        expanded_pool = list(chart_pool)
        for chart in support:
            if chart in _SUPPORTED_CHARTS and chart not in expanded_pool:
                expanded_pool.append(chart)

        for index, chart in enumerate(expanded_pool):
            heuristic_score, heuristic_reason = self._heuristic_score(chart, index, request)
            evidence = support.get(chart, [])
            retrieval_score = evidence[0].score if evidence else 0.0
            score = round(heuristic_score + retrieval_score + min(0.15, 0.03 * len(evidence)), 4)
            mapping = self._field_mapping(chart, request.data_profile, request.selected_fields)
            field_roles = self._field_roles_from_mapping(mapping, request.data_profile)
            spec_template = self._choose_spec_template(chart, evidence, mapping, field_roles)
            ranked.append(
                VisRAGCandidate(
                    candidate_id=f"candidate-{len(ranked) + 1:02d}-{chart}",
                    score=score,
                    chart_intent=self._chart_intent(chart),
                    recommended_mark=chart,
                    field_mapping=mapping,
                    field_roles=field_roles,
                    spec_template=spec_template,
                    transform_types=self._candidate_transform_types(chart, evidence),
                    support_example_ids=[item.example_id for item in evidence[:3]],
                    rationale=self._join_reason(heuristic_reason,
                                                f"{len(evidence)} retrieved example(s) support this chart family." if evidence else "No direct corpus support."),
                    caveats=[],
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.recommended_mark, item.candidate_id))
        return ranked

    def _heuristic_score(self, chart: str, index: int, request: VisRAGRequest) -> tuple[float, str]:
        profile = request.data_profile
        query_text = " ".join([request.query, request.intent, *request.operations]).lower()
        base = max(0.05, 0.2 - index * 0.02)
        if chart in {"line", "area"} and profile.temporal_columns and profile.numeric_columns:
            return base + 0.75, "Temporal field and numeric measure detected; trend chart is a strong fit."
        if chart == "point" and len(profile.numeric_columns) >= 2:
            return base + 0.70, "Two numeric fields detected; point/scatter chart fits relationship analysis."
        if chart in {"bar", "boxplot", "tick"} and profile.categorical_columns and profile.numeric_columns:
            return base + 0.62, "Categorical and numeric fields detected; comparison chart is a good fit."
        if chart == "histogram" and profile.numeric_columns:
            return base + 0.55, "Numeric measure detected; histogram fits distribution analysis."
        if chart in query_text or (chart == "point" and "scatter" in query_text):
            return base + 0.30, f"User language explicitly mentions or implies '{chart}'."
        return base, f"{chart} remains a fallback candidate based on the detected task."

    @staticmethod
    def _field_mapping(chart: str, profile: VisRAGDataProfile, selected_fields: list[str]) -> dict[str, str]:
        selected = [field for field in selected_fields if field in profile.column_names()]
        numeric = [field for field in selected if field in profile.numeric_columns] or profile.numeric_columns
        categorical = [field for field in selected if
                       field in profile.categorical_columns] or profile.categorical_columns
        temporal = [field for field in selected if field in profile.temporal_columns] or profile.temporal_columns
        if chart in {"line", "area"} and temporal and numeric:
            result = {"x": temporal[0], "y": numeric[0]}
            if categorical:
                result["color"] = categorical[0]
            return result
        if chart == "point" and len(numeric) >= 2:
            return {"x": numeric[0], "y": numeric[1]}
        if chart == "histogram" and numeric:
            return {"x": numeric[0], "y": numeric[0]}
        if chart in {"bar", "boxplot", "tick"} and categorical and numeric:
            return {"x": categorical[0], "y": numeric[0]}
        if numeric:
            return {"x": numeric[0], "y": numeric[1] if len(numeric) > 1 else numeric[0]}
        if categorical:
            return {"x": categorical[0], "y": categorical[0]}
        return {}

    @staticmethod
    def _field_roles_from_mapping(mapping: dict[str, str], profile: VisRAGDataProfile) -> dict[str, str]:
        roles: dict[str, str] = {}
        by_name = {column.name: column.semantic_type for column in profile.columns}
        for channel, field in mapping.items():
            semantic = by_name.get(field)
            if semantic == "datetime":
                roles[channel] = "temporal"
            elif semantic == "numeric":
                roles[channel] = "quantitative"
            else:
                roles[channel] = "nominal"
        return roles

    def _choose_spec_template(self, chart: str, evidence: list[Any], mapping: dict[str, str], roles: dict[str, str]) -> \
    dict[str, Any]:
        for example in evidence:
            if example.spec_template:
                template = deepcopy(example.spec_template)
                template.pop("data", None)
                template.pop("datasets", None)
                return template
        return self._default_spec_template(chart, mapping, roles)

    @staticmethod
    def _default_spec_template(chart: str, mapping: dict[str, str], roles: dict[str, str]) -> dict[str, Any]:
        mark = "point" if chart == "point" else chart
        spec: dict[str, Any] = {"mark": mark, "encoding": {}}
        if chart == "histogram" and mapping.get("x"):
            spec["encoding"]["x"] = {"field": mapping["x"], "type": roles.get("x", "quantitative"), "bin": True}
            spec["encoding"]["y"] = {"aggregate": "count", "type": "quantitative"}
            return spec
        for channel, field in mapping.items():
            spec["encoding"][channel] = {"field": field, "type": roles.get(channel, "nominal")}
        if chart in {"bar", "line", "area"} and mapping.get("y") and roles.get("y") == "quantitative":
            spec["encoding"].setdefault("y", {"field": mapping["y"], "type": "quantitative"})
            spec["encoding"]["y"].setdefault("aggregate", "mean")
        return spec

    @staticmethod
    def _candidate_transform_types(chart: str, evidence: list[Any]) -> list[str]:
        transforms: list[str] = []
        for example in evidence:
            for transform in example.transform_types:
                if transform not in transforms:
                    transforms.append(transform)
        if chart == "histogram" and "bin" not in transforms:
            transforms.append("bin")
        if chart in {"bar", "line", "area"} and "aggregate" not in transforms:
            transforms.append("aggregate")
        return transforms

    @staticmethod
    def _chart_intent(chart: str) -> str:
        return {
            "line": "temporal_trend",
            "area": "temporal_trend",
            "point": "relationship",
            "bar": "category_comparison",
            "histogram": "distribution",
            "boxplot": "distribution_comparison",
            "tick": "distribution",
        }.get(chart, "chart_generation")

    @staticmethod
    def _normalize_chart_candidates(values: list[str]) -> tuple[list[str], list[str]]:
        result: list[str] = []
        notes: list[str] = []
        seen: set[str] = set()
        for raw in values:
            chart = canonicalize_chart_type(raw)
            if raw and raw.strip().lower() in {"scatter", "scatterplot", "scatter plot", "scatter_chart"}:
                notes.append("Chart family 'scatter' is normalized to Vega-Lite 'point'.")
            if chart and chart in _SUPPORTED_CHARTS and chart not in seen:
                seen.add(chart)
                result.append(chart)
        return result, notes

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = value.strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                result.append(text)
        return result

    @staticmethod
    def _join_reason(*parts: str) -> str:
        return " ".join(part for part in parts if part).strip()
