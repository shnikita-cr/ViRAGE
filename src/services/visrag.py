from __future__ import annotations

from src.application.settings import ViRAGESettings
from src.domain.enums import ChartCaseType
from src.domain.models import DataProfile, QueryUnderstandingResult, VisRAGRecommendation, VisRAGResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.visrag_retrieval import (
    canonicalize_chart_type,
    retrieve_examples,
    summarize_chart_support,
)

_SUPPORTED_PIPELINE_CHARTS = {"line", "bar", "scatter", "histogram", "boxplot"}
_FALLBACK_CHART_MAP = {
    "area": "line",
    "heatmap": "bar",
}


class VisRAGService(BaseService):
    def invoke(
            self,
            query_understanding: QueryUnderstandingResult,
            data_profile: DataProfile,
            runtime: RuntimeContext,
    ) -> VisRAGResult:
        candidate_charts, normalization_notes = self._normalize_candidate_charts(query_understanding.candidate_charts)
        if not candidate_charts:
            candidate_charts = ["bar", "line"]

        retrieval_query = self._build_retrieval_query(query_understanding, data_profile)
        retrieved = retrieve_examples(
            runtime.settings.visrag_corpus_root,
            query_text=retrieval_query,
            preferred_chart_types=candidate_charts,
            top_k=runtime.settings.visrag_top_k_examples,
            min_score=runtime.settings.visrag_min_example_score,
        )
        support = summarize_chart_support(retrieved.examples)

        recommendations = self._build_recommendations(
            query_understanding=query_understanding,
            data_profile=data_profile,
            candidate_charts=candidate_charts,
            support=support,
            settings=runtime.settings,
        )

        caveats = list(query_understanding.constraints)
        caveats.extend(normalization_notes)
        if query_understanding.case_type is ChartCaseType.NON_CANONICAL:
            caveats.append("Non-canonical case requires conservative interpretation.")
        if not retrieved.examples:
            caveats.append("No matching local corpus examples were retrieved; heuristic ranking was used.")

        return VisRAGResult(
            recommendations=recommendations,
            rules=[
                "Use clear titles and axis labels.",
                "Avoid overcrowded visuals.",
                "Prefer readable defaults.",
                "Prefer examples retrieved from the local corpus when they agree with the data shape.",
            ],
            caveats=caveats,
            retrieved_examples=[item.to_model() for item in retrieved.examples],
            corpus_status=retrieved.corpus_status,
            retrieval_strategy="hybrid_plot2code_plus_heuristics" if retrieved.examples else "heuristic_only",
        )

    def _build_recommendations(
            self,
            *,
            query_understanding: QueryUnderstandingResult,
            data_profile: DataProfile,
            candidate_charts: list[str],
            support: dict[str, list],
            settings: ViRAGESettings,
    ) -> list[VisRAGRecommendation]:
        chart_pool = list(candidate_charts)
        for chart_type in support.keys():
            if chart_type not in chart_pool and chart_type in _SUPPORTED_PIPELINE_CHARTS:
                chart_pool.append(chart_type)

        scored: list[tuple[str, float, str, list[str]]] = []
        for index, chart_type in enumerate(chart_pool):
            heuristic_score, heuristic_reason = self._heuristic_score(
                chart_type=chart_type,
                index=index,
                query_understanding=query_understanding,
                data_profile=data_profile,
            )
            evidence = support.get(chart_type, [])
            retrieval_score = self._retrieval_score(evidence)
            total_score = round(heuristic_score + retrieval_score, 4)

            rationale_parts = [heuristic_reason]
            if evidence:
                rationale_parts.append(
                    f"{len(evidence)} local corpus match(es) from Plot2Code support this chart family."
                )
            rationale = " ".join(part for part in rationale_parts if part)
            support_examples = [item.example_id for item in evidence[:3]]

            scored.append((chart_type, total_score, rationale, support_examples))

        scored.sort(key=lambda item: (-item[1], item[0]))

        recommendations: list[VisRAGRecommendation] = []
        for priority, (chart_type, score, rationale, support_examples) in enumerate(
                scored[: settings.visrag_top_k_recommendations],
                start=1,
        ):
            recommendations.append(
                VisRAGRecommendation(
                    chart_family=chart_type,
                    rationale=rationale,
                    priority=priority,
                    score=score,
                    support_examples=support_examples,
                )
            )
        return recommendations

    def _heuristic_score(
            self,
            *,
            chart_type: str,
            index: int,
            query_understanding: QueryUnderstandingResult,
            data_profile: DataProfile,
    ) -> tuple[float, str]:
        base = max(0.10, 0.40 - index * 0.05)
        ops_text = " ".join(query_understanding.requested_operations).lower()

        if chart_type == "line":
            if data_profile.likely_time_columns and data_profile.likely_numeric_columns:
                return base + 0.60, "Time-like field and numeric field detected; line chart suits trend analysis."
            return base + 0.15, "Line chart kept as a general-purpose fallback for trend-like queries."

        if chart_type == "scatter":
            if len(data_profile.likely_numeric_columns) >= 2:
                return base + 0.55, "Two numeric fields detected; scatter suits relationship analysis."
            return base + 0.10, "Scatter kept as a weak fallback for relationship-style requests."

        if chart_type == "histogram":
            if data_profile.likely_numeric_columns:
                return base + 0.45, "Numeric field detected; histogram suits distribution analysis."
            return base + 0.05, "Histogram retained, but numeric evidence is weak."

        if chart_type == "boxplot":
            if data_profile.likely_categorical_columns and data_profile.likely_numeric_columns:
                return base + 0.45, "Categorical and numeric fields detected; boxplot suits grouped spread analysis."
            return base + 0.08, "Boxplot retained, but grouping evidence is weak."

        if chart_type == "bar":
            if data_profile.likely_categorical_columns and data_profile.likely_numeric_columns:
                return base + 0.45, "Categorical and numeric fields detected; bar chart suits comparison analysis."
            if "comparison" in ops_text or "ranking" in ops_text:
                return base + 0.25, "Bar chart suits comparison or ranking requests."
            return base + 0.15, "Bar chart kept as the safest default for the current pipeline."

        return base, f"{chart_type} retained as a low-priority fallback."

    @staticmethod
    def _retrieval_score(evidence: list) -> float:
        if not evidence:
            return 0.0
        top_scores = [item.score for item in evidence[:3]]
        return round(sum(top_scores) / len(top_scores), 4)

    @staticmethod
    def _build_retrieval_query(query_understanding: QueryUnderstandingResult, data_profile: DataProfile) -> str:
        parts: list[str] = [
            query_understanding.intent,
            " ".join(query_understanding.requested_operations),
            " ".join(query_understanding.constraints),
        ]
        if data_profile.likely_time_columns:
            parts.append("time series temporal trend")
        if len(data_profile.likely_numeric_columns) >= 2:
            parts.append("numeric relationship comparison")
        elif data_profile.likely_numeric_columns:
            parts.append("single numeric measure")
        if data_profile.likely_categorical_columns:
            parts.append("categorical grouping")
        return " ".join(part for part in parts if part).strip()

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
            elif not chart:
                chart = raw if raw in _SUPPORTED_PIPELINE_CHARTS else ""

            if raw in _FALLBACK_CHART_MAP:
                notes.append(
                    f"Chart family '{raw}' is not directly supported by the current renderer; using '{chart}' fallback."
                )

            if chart and chart in _SUPPORTED_PIPELINE_CHARTS and chart not in seen:
                seen.add(chart)
                normalized.append(chart)

        return normalized, notes
