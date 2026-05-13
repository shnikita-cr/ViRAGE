from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.domain.models import CandidateSpec, SpecGenerationAttempt, SpecGenerationRequest, SpecGenerationResult
from src.infrastructure.runtime import RuntimeContext
from src.services.spec_generation.base import SpecGenerationBackend
from src.visrag_core import canonicalize_chart_type, normalize_aggregate

_POSITION_CHANNELS = {"x", "y"}

_NON_POSITION_CHANNELS_WITHOUT_AXIS = {
    "color",
    "size",
    "shape",
    "opacity",
    "text",
    "tooltip",
    "detail",
    "row",
    "column",
    "theta",
    "radius",
    "longitude",
    "latitude",
}


class TemplateSpecBackend(SpecGenerationBackend):
    """Legacy backend: materialize the selected RAG spec_template."""

    backend_name = "template"

    def generate(self, request: SpecGenerationRequest, runtime: RuntimeContext) -> SpecGenerationResult:
        selected = request.candidate_spec_set.selected_candidate_spec
        if selected is None:
            raise RuntimeError("Template spec generation requires a selected RAG candidate.")

        spec = self._build_spec(request, selected)
        result = SpecGenerationResult(
            backend_name=self.backend_name,
            prompt_version="legacy_template",
            spec_json=spec,
            spec_without_runtime_data=self._without_runtime_data(spec),
            explanation=selected.rationale or selected.summary,
            attempts=[
                SpecGenerationAttempt(
                    attempt_number=1,
                    status="succeeded",
                    explanation=selected.rationale or selected.summary,
                    spec_json=spec,
                )
            ],
            used_visrag_context=True,
            generation_attempt_number=request.generation_attempt_number,
            max_generation_attempts=request.max_generation_attempts,
            previous_validation_errors=list(request.previous_validation_errors),
            previous_repair_hints=list(request.previous_repair_hints),
        )
        artifact_paths = self._save_result_artifacts(runtime, result)
        return result.model_copy(update={"artifact_paths": artifact_paths})

    def _build_spec(self, request: SpecGenerationRequest, candidate: CandidateSpec) -> dict[str, Any]:
        if not candidate.spec_template:
            raise RuntimeError(
                f"RAG candidate {candidate.spec_id!r} has no Vega-Lite spec_template."
            )

        spec = deepcopy(candidate.spec_template)

        if not isinstance(spec, dict) or not spec:
            raise RuntimeError(
                f"RAG candidate {candidate.spec_id!r} spec_template must be a non-empty object."
            )

        spec["$schema"] = spec.get("$schema") or "https://vega.github.io/schema/vega-lite/v5.json"
        spec["data"] = {"url": request.prepared.output_path}
        spec["mark"] = self._normalize_mark(spec.get("mark") or candidate.chart_family)
        spec.setdefault("title", candidate.summary)
        spec.setdefault("description", candidate.rationale or candidate.summary)

        encoding = spec.get("encoding")

        if not isinstance(encoding, dict) or not encoding:
            raise RuntimeError(
                f"RAG candidate {candidate.spec_id!r} spec_template must contain encoding."
            )

        self._normalize_encoding(encoding)
        self._apply_safe_field_mapping(spec, request.prepared.column_name_map)

        return spec

    @classmethod
    def _apply_safe_field_mapping(
            cls,
            value: Any,
            column_name_map: dict[str, str],
    ) -> None:
        if not column_name_map:
            return

        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key == "field" and isinstance(item, str):
                    value[key] = column_name_map.get(item, item)
                    continue

                if key in {"groupby", "fields"} and isinstance(item, list):
                    value[key] = [column_name_map.get(entry, entry) if isinstance(entry, str) else entry for entry in
                                  item]
                    continue

                cls._apply_safe_field_mapping(item, column_name_map)
            return

        if isinstance(value, list):
            for item in value:
                cls._apply_safe_field_mapping(item, column_name_map)

    @staticmethod
    def _normalize_mark(mark: Any) -> Any:
        if isinstance(mark, dict):
            clone = dict(mark)
            clone["type"] = canonicalize_chart_type(str(clone.get("type") or "bar"))
            return clone

        return canonicalize_chart_type(str(mark or "bar"))

    @classmethod
    def _normalize_encoding(cls, encoding: dict[str, Any]) -> None:
        for channel, channel_spec in encoding.items():
            if isinstance(channel_spec, list):
                for item in channel_spec:
                    if isinstance(item, dict):
                        cls._normalize_channel(channel, item)

            elif isinstance(channel_spec, dict):
                cls._normalize_channel(channel, channel_spec)

    @classmethod
    def _normalize_channel(
            cls,
            channel: str,
            channel_spec: dict[str, Any],
    ) -> None:
        aggregate = normalize_aggregate(channel_spec.get("aggregate"))

        if aggregate:
            channel_spec["aggregate"] = aggregate

        if channel in _POSITION_CHANNELS:
            cls._normalize_position_axis(channel_spec)
            return

        if channel in _NON_POSITION_CHANNELS_WITHOUT_AXIS:
            channel_spec.pop("axis", None)

    @staticmethod
    def _normalize_position_axis(channel_spec: dict[str, Any]) -> None:
        axis = channel_spec.setdefault("axis", {})

        if not isinstance(axis, dict):
            channel_spec["axis"] = {}
            axis = channel_spec["axis"]

        axis.setdefault("labelLimit", 180)
        axis.setdefault("labelOverlap", "greedy")

        if channel_spec.get("type") == "temporal":
            axis.setdefault("format", "%Y-%m-%d")
            axis.setdefault("labelAngle", -35)

    @staticmethod
    def _without_runtime_data(spec: dict[str, Any]) -> dict[str, Any]:
        clone = deepcopy(spec)
        clone.pop("data", None)
        clone.pop("datasets", None)
        return clone

    @staticmethod
    def _save_result_artifacts(runtime: RuntimeContext, result: SpecGenerationResult) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        if runtime.current_run_id:
            attempt_prefix = f"spec_generation_attempt_{result.generation_attempt_number:03d}"
            artifacts[f"{attempt_prefix}_result"] = runtime.save_json_artifact(
                f"artifacts/{attempt_prefix}_result.json",
                result.model_dump(exclude={"artifact_paths"}),
                numbered=True,
            )
        return artifacts
