from __future__ import annotations

from src.domain.enums import ChartCaseType
from src.domain.models import DataProfile, QueryUnderstandingResult, VisRAGRecommendation, VisRAGResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.visrag_retrieval import VisRAGRetriever


class VisRAGService(BaseService):
    def __init__(self) -> None:
        self.retriever = VisRAGRetriever()

    def invoke(
            self,
            query_understanding: QueryUnderstandingResult,
            data_profile: DataProfile,
            runtime: RuntimeContext,
    ) -> VisRAGResult:
        retrieved_examples, corpus_status = self.retriever.retrieve(
            corpus_root=runtime.settings.visrag_corpus_root,
            query_understanding=query_understanding,
            data_profile=data_profile,
            top_k=runtime.settings.visrag_top_k_examples,
        )
        recommendations = self._build_recommendations(
            query_understanding=query_understanding,
            data_profile=data_profile,
            retrieved_examples=retrieved_examples,
            top_k=runtime.settings.visrag_top_k_recommendations,
        )
        return VisRAGResult(
            recommendations=recommendations,
            rules=self._build_rules(query_understanding),
            caveats=self._build_caveats(query_understanding, data_profile, corpus_status),
            retrieved_examples=retrieved_examples,
            corpus_status=corpus_status,
            retrieval_strategy="hybrid_rule_retrieval",
        )

    def _build_recommendations(
            self,
            *,
            query_understanding: QueryUnderstandingResult,
            data_profile: DataProfile,
            retrieved_examples,
            top_k: int,
    ) -> list[VisRAGRecommendation]:
        candidate_families = self._ordered_candidate_families(query_understanding, retrieved_examples)
        scored = []
        for family in candidate_families:
            support = [item for item in retrieved_examples if item.chart_family == family]
            score = self._heuristic_fit(family, data_profile)
            score += self._candidate_priority_bonus(family, query_understanding.candidate_charts)
            score += sum(item.score for item in support[:2])
            rationale_parts = []
            fit_reason = self._fit_reason(family, data_profile, query_understanding)
            if fit_reason:
                rationale_parts.append(fit_reason)
            if support:
                corpora = ", ".join(self._dedupe([item.corpus for item in support[:2]]))
                rationale_parts.append(f"Retrieved similar examples from {corpora}.")
            if not rationale_parts:
                rationale_parts.append(
                    f"Suggested for {', '.join(query_understanding.requested_operations) or 'exploratory analysis'}."
                )
            scored.append(
                VisRAGRecommendation(
                    chart_family=family,
                    rationale=" ".join(rationale_parts),
                    priority=0,
                    score=round(score, 4),
                    supporting_example_ids=[item.example_id for item in support[:2]],
                    supporting_corpora=self._dedupe([item.corpus for item in support[:2]]),
                )
            )

        ranked = sorted(scored, key=lambda item: (-item.score, item.chart_family))[:top_k]
        for idx, item in enumerate(ranked, start=1):
            item.priority = idx
        return ranked

    @staticmethod
    def _ordered_candidate_families(query_understanding: QueryUnderstandingResult, retrieved_examples) -> list[str]:
        families = list(query_understanding.candidate_charts)
        families.extend(example.chart_family for example in retrieved_examples if example.chart_family != "unknown")
        if not families:
            families = ["bar", "line"]
        return VisRAGService._dedupe([family.lower() for family in families if family])

    @staticmethod
    def _candidate_priority_bonus(family: str, candidates: list[str]) -> float:
        normalized = [item.lower() for item in candidates]
        if family not in normalized:
            return 0.0
        return max(0.0, 1.2 - (normalized.index(family) * 0.2))

    @staticmethod
    def _heuristic_fit(family: str, data_profile: DataProfile) -> float:
        numeric = len(data_profile.likely_numeric_columns)
        categorical = len(data_profile.likely_categorical_columns)
        time_like = len(data_profile.likely_time_columns)
        if family == "line":
            return 1.2 if time_like and numeric else 0.25
        if family == "scatter":
            return 1.1 if numeric >= 2 else 0.1
        if family == "histogram":
            return 0.9 if numeric >= 1 else 0.1
        if family == "boxplot":
            return 0.95 if numeric >= 1 and categorical >= 1 else 0.15
        if family == "bar":
            return 0.85 if numeric >= 1 and (categorical >= 1 or time_like >= 1) else 0.2
        if family == "heatmap":
            return 0.7 if numeric >= 2 else 0.1
        return 0.2

    @staticmethod
    def _fit_reason(
            family: str,
            data_profile: DataProfile,
            query_understanding: QueryUnderstandingResult,
    ) -> str:
        operations = ", ".join(query_understanding.requested_operations) or "the request"
        if family == "line" and data_profile.likely_time_columns and data_profile.likely_numeric_columns:
            return "Time-like and numeric fields are available; line chart fits trend analysis well."
        if family == "scatter" and len(data_profile.likely_numeric_columns) >= 2:
            return "At least two numeric fields are available; scatter suits relationship analysis."
        if family == "histogram" and data_profile.likely_numeric_columns:
            return "Numeric fields are available; histogram is suitable for distribution analysis."
        if family == "boxplot" and data_profile.likely_numeric_columns and data_profile.likely_categorical_columns:
            return "Numeric and categorical fields are available; boxplot supports grouped spread analysis."
        if family == "bar" and data_profile.likely_numeric_columns:
            return f"Bar chart remains a readable default for {operations}."
        return "Selected as a conservative fallback based on the request and available data shape."

    @staticmethod
    def _build_rules(query_understanding: QueryUnderstandingResult) -> list[str]:
        rules = [
            "Use clear titles and axis labels.",
            "Avoid overcrowded visuals.",
            "Prefer readable defaults.",
            "Prefer chart families supported by both the data profile and retrieved reference examples.",
        ]
        if query_understanding.case_type is ChartCaseType.NON_CANONICAL:
            rules.append("Validate whether a simpler canonical chart can communicate the same message.")
        return rules

    @staticmethod
    def _build_caveats(
            query_understanding: QueryUnderstandingResult,
            data_profile: DataProfile,
            corpus_status: list[str],
    ) -> list[str]:
        caveats = list(query_understanding.constraints)
        if query_understanding.case_type is ChartCaseType.NON_CANONICAL:
            caveats.append("Non-canonical case requires conservative interpretation.")
        if not data_profile.likely_numeric_columns:
            caveats.append("No numeric columns detected; chart choices may be limited.")
        if not data_profile.likely_time_columns and "line" in [item.lower() for item in
                                                               query_understanding.candidate_charts]:
            caveats.append("Line chart was requested without a detected time-like field.")
        caveats.extend(status for status in corpus_status if "no local normalized corpus" in status.lower())
        return VisRAGService._dedupe(caveats)

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
