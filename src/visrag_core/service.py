from __future__ import annotations

import re
from copy import deepcopy

from .chart_types import canonicalize_chart_type, normalize_aggregate, require_supported_chart_type
from .corpus import VisRAGCorpus
from .models import VisRAGCandidate, VisRAGConfig, VisRAGRequest, VisRAGResult, VisRAGExample

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class VisRAGCoreService:
    def __init__(self, config: VisRAGConfig | None = None):
        self.config = config or VisRAGConfig()
        self._examples: list[VisRAGExample] | None = None

    def search(self, request: VisRAGRequest) -> VisRAGResult:
        preferred = {require_supported_chart_type(item) for item in request.preferred_chart_types if item}
        examples = self._load_examples()
        scored = [candidate for example in examples if (candidate := self._score_example(example, request, preferred))]
        scored.sort(key=lambda item: (-item.score, item.example.chart_type, item.example.example_id))
        return VisRAGResult(
            query=request.query,
            candidates=scored[: request.top_k],
            corpus_root=self.config.corpus_root.as_posix(),
        )

    def _load_examples(self) -> list[VisRAGExample]:
        if self._examples is None:
            self._examples = VisRAGCorpus(self.config.corpus_root).load()
        return self._examples

    def _score_example(
            self,
            example: VisRAGExample,
            request: VisRAGRequest,
            preferred: set[str],
    ) -> VisRAGCandidate | None:
        chart_type = canonicalize_chart_type(example.chart_type)
        if preferred and chart_type not in preferred:
            return None
        field_mapping = self._map_fields(example.field_roles, request)
        if example.field_roles and len(field_mapping) < len(example.field_roles):
            return None
        score = self._text_score(example, request.query)
        if preferred and chart_type in preferred:
            score += 1.0
        score += 0.25 * len(field_mapping)
        if score <= 0:
            return None
        return VisRAGCandidate(
            example=example,
            score=round(score, 4),
            field_mapping=field_mapping,
            spec_template=self._materialize_template(example, field_mapping),
        )

    @staticmethod
    def _text_score(example: VisRAGExample, query: str) -> float:
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        if not query_tokens:
            return 0.0
        corpus_tokens = set(
            _TOKEN_RE.findall(" ".join([example.instruction, example.description or "", *example.keywords]).lower()))
        return len(query_tokens & corpus_tokens) / max(1, len(query_tokens))

    @staticmethod
    def _map_fields(field_roles: dict[str, str], request: VisRAGRequest) -> dict[str, str]:
        mapping: dict[str, str] = {}
        used: set[str] = set()
        selected = [name for name in request.selected_fields if name]
        columns = request.data_profile.columns
        for channel, role in field_roles.items():
            role_key = _normalize_role(role)
            candidates = [column.name for column in columns if _normalize_role(column.semantic_type) == role_key]
            ordered = [name for name in selected if name in candidates] + [name for name in candidates if
                                                                           name not in selected]
            chosen = next((name for name in ordered if name not in used), None)
            if chosen:
                mapping[channel] = chosen
                used.add(chosen)
        return mapping

    @staticmethod
    def _materialize_template(example: VisRAGExample, field_mapping: dict[str, str]) -> dict:
        if not field_mapping:
            return deepcopy(example.spec_template)
        encoding: dict[str, dict[str, str]] = {}
        for channel, field_name in field_mapping.items():
            role = example.field_roles.get(channel, "nominal")
            encoding[channel] = {"field": field_name, "type": _vega_type(role)}
            aggregate = _find_channel_aggregate(example.spec_template, channel)
            if aggregate:
                encoding[channel]["aggregate"] = normalize_aggregate(aggregate)
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": canonicalize_chart_type(example.chart_type),
            "encoding": encoding,
        }


def _normalize_role(value: str | None) -> str:
    text = (value or "").lower()
    if text in {"numeric", "number", "measure", "quantitative"}:
        return "quantitative"
    if text in {"date", "time", "datetime", "temporal", "year"}:
        return "temporal"
    return "nominal"


def _vega_type(value: str | None) -> str:
    return _normalize_role(value)


def _find_channel_aggregate(spec: dict, channel: str) -> str | None:
    encoding = spec.get("encoding") if isinstance(spec, dict) else None
    channel_spec = encoding.get(channel) if isinstance(encoding, dict) else None
    if isinstance(channel_spec, dict):
        aggregate = channel_spec.get("aggregate")
        return str(aggregate) if aggregate else None
    return None
