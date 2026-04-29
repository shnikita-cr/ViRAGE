from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.domain.models import CandidateSpec, CandidateSpecSet, DataPreparationResult, VegaLiteSpecArtifact
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.visrag_core import canonicalize_chart_type, normalize_aggregate


class ChartGeneratorService(BaseService):
    """Create a Vega-Lite artifact from the selected RAG candidate."""

    def invoke(
            self,
            prepared: DataPreparationResult,
            candidate_spec_set: CandidateSpecSet,
            runtime: RuntimeContext,
    ) -> VegaLiteSpecArtifact:
        selected = candidate_spec_set.selected_candidate_spec
        if selected is None:
            raise RuntimeError("Chart generation requires a selected RAG candidate.")
        return VegaLiteSpecArtifact(spec_json=self._build_spec(prepared, selected), version="v1")

    async def ainvoke(
            self,
            prepared: DataPreparationResult,
            candidate_spec_set: CandidateSpecSet,
            runtime: RuntimeContext,
    ) -> VegaLiteSpecArtifact:
        return self.invoke(prepared, candidate_spec_set, runtime)

    @classmethod
    def _build_spec(cls, prepared: DataPreparationResult, candidate: CandidateSpec) -> dict[str, Any]:
        if not candidate.spec_template:
            raise RuntimeError(f"RAG candidate {candidate.spec_id!r} has no Vega-Lite spec_template.")
        spec = deepcopy(candidate.spec_template)
        if not isinstance(spec, dict) or not spec:
            raise RuntimeError(f"RAG candidate {candidate.spec_id!r} spec_template must be a non-empty object.")
        spec["$schema"] = spec.get("$schema") or "https://vega.github.io/schema/vega-lite/v5.json"
        spec["data"] = {"url": prepared.output_path}
        spec["mark"] = cls._normalize_mark(spec.get("mark") or candidate.chart_family)
        spec.setdefault("title", candidate.summary)
        spec.setdefault("description", candidate.rationale or candidate.summary)
        encoding = spec.get("encoding")
        if not isinstance(encoding, dict) or not encoding:
            raise RuntimeError(f"RAG candidate {candidate.spec_id!r} spec_template must contain encoding.")
        cls._normalize_encoding(encoding)
        return spec

    @staticmethod
    def _normalize_mark(mark: Any) -> Any:
        if isinstance(mark, dict):
            clone = dict(mark)
            clone["type"] = canonicalize_chart_type(str(clone.get("type") or "bar"))
            return clone
        return canonicalize_chart_type(str(mark or "bar"))

    @classmethod
    def _normalize_encoding(cls, encoding: dict[str, Any]) -> None:
        for channel_spec in encoding.values():
            if isinstance(channel_spec, list):
                for item in channel_spec:
                    if isinstance(item, dict):
                        cls._normalize_channel(item)
            elif isinstance(channel_spec, dict):
                cls._normalize_channel(channel_spec)

    @staticmethod
    def _normalize_channel(channel_spec: dict[str, Any]) -> None:
        aggregate = normalize_aggregate(channel_spec.get("aggregate"))
        if aggregate:
            channel_spec["aggregate"] = aggregate
        axis = channel_spec.setdefault("axis", {})
        if not isinstance(axis, dict):
            return
        axis.setdefault("labelLimit", 180)
        axis.setdefault("labelOverlap", "greedy")
        if channel_spec.get("type") == "temporal":
            axis.setdefault("format", "%Y-%m-%d")
            axis.setdefault("labelAngle", -35)
