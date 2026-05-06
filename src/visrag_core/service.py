from __future__ import annotations

from .chart_types import SUPPORTED_CHART_TYPES, canonicalize_chart_type
from .corpus import VisRAGCorpus
from .field_grounding import map_fields
from .models import VisRAGCandidate, VisRAGConfig, VisRAGExample, VisRAGRequest, VisRAGResult
from .retrievers import build_retriever
from .spec_materializer import materialize_spec_template


class VisRAGCoreService:
    def __init__(self, config: VisRAGConfig | None = None):
        self.config = config or VisRAGConfig()
        self._examples: list[VisRAGExample] | None = None

    def search(self, request: VisRAGRequest) -> VisRAGResult:
        caveats: list[str] = []
        examples = self._load_examples(caveats)
        if not examples:
            return self._result(request, [], caveats + ["empty_result: no corpus examples are available."])

        preferred = self._preferred_chart_types(request.preferred_chart_types, caveats)
        retrieved = build_retriever(self.config).search(request, examples)
        candidates, failed_mappings = self._rank_candidates(retrieved, request, preferred)
        if not candidates and failed_mappings == 0:
            failed_mappings = self._count_possible_failed_mappings(examples, request, preferred)
        if failed_mappings:
            caveats.append(
                f"failed_field_mapping: {failed_mappings} retrieved examples were incompatible with the data profile.")
        if not candidates:
            caveats.append("empty_result: no compatible RAG candidates were found.")
        return self._result(request, candidates[: request.top_k], caveats)

    def _load_examples(self, caveats: list[str]) -> list[VisRAGExample]:
        if self._examples is not None:
            return self._examples
        try:
            self._examples = VisRAGCorpus(self.config.corpus_root).load()
        except FileNotFoundError:
            caveats.append(f"missing_corpus: {self.config.corpus_root.as_posix()} does not exist.")
            self._examples = []
        return self._examples

    @staticmethod
    def _preferred_chart_types(values: list[str], caveats: list[str]) -> set[str]:
        result: set[str] = set()
        for value in values:
            chart_type = canonicalize_chart_type(value)
            if not chart_type:
                continue
            if chart_type not in SUPPORTED_CHART_TYPES:
                caveats.append(f"unsupported_chart_type: {value!r}.")
                continue
            result.add(chart_type)
        return result

    @staticmethod
    def _rank_candidates(
            retrieved: list[VisRAGCandidate],
            request: VisRAGRequest,
            preferred: set[str],
    ) -> tuple[list[VisRAGCandidate], int]:
        result: list[VisRAGCandidate] = []
        failed_mappings = 0
        for candidate in retrieved:
            chart_type = canonicalize_chart_type(candidate.example.chart_type)
            if preferred and chart_type not in preferred:
                continue
            field_mapping, missing_channels = map_fields(candidate.example.field_roles, request)
            if missing_channels:
                failed_mappings += 1
                continue
            field_score = 0.25 * len(field_mapping)
            chart_score = 1.0 if preferred and chart_type in preferred else 0.0
            text_score = float(candidate.score)
            total = text_score + chart_score + field_score
            candidate.score = round(total, 6)
            candidate.confidence = max(0.0, min(1.0, total / 2.5))
            candidate.score_breakdown = {
                **candidate.score_breakdown,
                "text": round(text_score, 6),
                "chart_type": chart_score,
                "field_mapping": round(field_score, 6),
                "total": candidate.score,
            }
            candidate.field_mapping = field_mapping
            candidate.spec_template = materialize_spec_template(candidate.example, field_mapping)
            result.append(candidate)
        result.sort(key=lambda item: (-item.score, item.example.chart_type, item.example.example_id))
        return result, failed_mappings

    @staticmethod
    def _count_possible_failed_mappings(
            examples: list[VisRAGExample],
            request: VisRAGRequest,
            preferred: set[str],
    ) -> int:
        failed = 0
        for example in examples:
            chart_type = canonicalize_chart_type(example.chart_type)
            if preferred and chart_type not in preferred:
                continue
            _, missing_channels = map_fields(example.field_roles, request)
            if missing_channels:
                failed += 1
        return failed

    def _result(self, request: VisRAGRequest, candidates: list[VisRAGCandidate], caveats: list[str]) -> VisRAGResult:
        return VisRAGResult(
            query=request.query,
            candidates=candidates,
            corpus_root=self.config.corpus_root.as_posix(),
            caveats=_dedupe(caveats),
        )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result
