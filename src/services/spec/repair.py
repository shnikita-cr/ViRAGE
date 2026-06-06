from __future__ import annotations

from copy import deepcopy
from typing import Any

VEGA_LITE_SCHEMA_URL = "https://vega.github.io/schema/vega-lite/v5.json"


class SpecRepairService:
    """Small deterministic repairs for common LLM-generated Vega-Lite mistakes.

    The service is intentionally conservative: it only changes structures that are clearly invalid or known aliases.
    It never guesses missing fields and never replaces the chart intent selected by the generator.
    """

    def repair(self, spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        repaired = deepcopy(spec)
        notes: list[str] = []

        if not isinstance(repaired, dict):
            return {}, ["Specification was not a JSON object and could not be repaired."]

        if "$schema" not in repaired:
            repaired["$schema"] = VEGA_LITE_SCHEMA_URL
            notes.append("Added missing Vega-Lite v5 schema.")

        self._normalize_scatter_alias(repaired, notes)
        self._repair_view_level_facet_without_spec(repaired, notes)
        self._repair_string_channels(repaired, notes)
        self._remove_empty_or_null_scale_domains(repaired, notes)
        self._remove_invalid_transform_items(repaired, notes)
        self._remove_empty_top_level_datasets(repaired, notes)
        return repaired, self._dedupe(notes)

    @classmethod
    def _normalize_scatter_alias(cls, node: Any, notes: list[str]) -> None:
        if isinstance(node, dict):
            mark = node.get("mark")
            if isinstance(mark, str) and mark.strip().lower() == "scatter":
                node["mark"] = "point"
                notes.append("Normalized non-standard mark 'scatter' to Vega-Lite mark 'point'.")
            elif isinstance(mark, dict):
                mark_type = mark.get("type")
                if isinstance(mark_type, str) and mark_type.strip().lower() == "scatter":
                    mark["type"] = "point"
                    notes.append("Normalized non-standard mark type 'scatter' to Vega-Lite mark 'point'.")
            for value in node.values():
                cls._normalize_scatter_alias(value, notes)
        elif isinstance(node, list):
            for item in node:
                cls._normalize_scatter_alias(item, notes)

    @classmethod
    def _repair_view_level_facet_without_spec(cls, node: Any, notes: list[str]) -> None:
        if isinstance(node, dict):
            facet = node.get("facet")
            if isinstance(facet, str) and "mark" in node:
                node.setdefault("encoding", {})["column"] = {"field": facet, "type": "nominal"}
                node.pop("facet", None)
                notes.append("Moved string facet into encoding.column because view-level facet was incomplete.")
            elif isinstance(facet, dict) and "spec" not in node and "mark" in node:
                encoding = node.setdefault("encoding", {})
                moved = False
                for source_key, target_key in (("row", "row"), ("column", "column"), ("field", "column")):
                    value = facet.get(source_key)
                    if value is None:
                        continue
                    encoding[target_key] = cls._channel_def(value)
                    moved = True
                if moved:
                    node.pop("facet", None)
                    notes.append("Moved incomplete view-level facet definition into encoding row/column channels.")
            for value in list(node.values()):
                cls._repair_view_level_facet_without_spec(value, notes)
        elif isinstance(node, list):
            for item in node:
                cls._repair_view_level_facet_without_spec(item, notes)

    @classmethod
    def _repair_string_channels(cls, node: Any, notes: list[str]) -> None:
        if isinstance(node, dict):
            encoding = node.get("encoding")
            if isinstance(encoding, dict):
                for channel, value in list(encoding.items()):
                    if isinstance(value, str) and value.strip():
                        encoding[channel] = {"field": value.strip(), "type": cls._default_type_for_channel(channel)}
                        notes.append(f"Converted string encoding.{channel} into a Vega-Lite channel definition.")
            for value in node.values():
                cls._repair_string_channels(value, notes)
        elif isinstance(node, list):
            for item in node:
                cls._repair_string_channels(item, notes)

    @classmethod
    def _remove_empty_or_null_scale_domains(cls, node: Any, notes: list[str]) -> None:
        if isinstance(node, dict):
            scale = node.get("scale")
            if isinstance(scale, dict) and "domain" in scale:
                domain = scale.get("domain")
                if domain is None:
                    scale.pop("domain", None)
                    notes.append("Removed null scale.domain.")
                elif isinstance(domain, list):
                    cleaned = [item for item in domain if item is not None]
                    if len(cleaned) != len(domain):
                        if cleaned:
                            scale["domain"] = cleaned
                            notes.append("Removed null values from scale.domain.")
                        else:
                            scale.pop("domain", None)
                            notes.append("Removed empty scale.domain after null cleanup.")
            for value in node.values():
                cls._remove_empty_or_null_scale_domains(value, notes)
        elif isinstance(node, list):
            for item in node:
                cls._remove_empty_or_null_scale_domains(item, notes)

    @classmethod
    def _remove_invalid_transform_items(cls, node: Any, notes: list[str]) -> None:
        if isinstance(node, dict):
            transform = node.get("transform")
            if isinstance(transform, list):
                cleaned = [item for item in transform if cls._is_valid_transform_item(item)]
                if len(cleaned) != len(transform):
                    node["transform"] = cleaned
                    notes.append("Removed empty or unsupported transform entries.")
            for value in node.values():
                cls._remove_invalid_transform_items(value, notes)
        elif isinstance(node, list):
            for item in node:
                cls._remove_invalid_transform_items(item, notes)

    @staticmethod
    def _remove_empty_top_level_datasets(node: dict[str, Any], notes: list[str]) -> None:
        datasets = node.get("datasets")
        if isinstance(datasets, dict) and not datasets:
            node.pop("datasets", None)
            notes.append("Removed empty top-level datasets object; runtime data.url is used instead.")

    @staticmethod
    def _is_valid_transform_item(item: Any) -> bool:
        if not isinstance(item, dict) or not item:
            return False
        allowed_keys = {
            "aggregate", "bin", "calculate", "density", "extent", "filter", "flatten", "fold", "impute",
            "joinaggregate", "lookup", "pivot", "quantile", "regression", "sample", "stack", "timeUnit", "window",
        }
        return bool(allowed_keys.intersection(item.keys()))

    @staticmethod
    def _channel_def(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            result = dict(value)
            result.setdefault("type", "nominal")
            return result
        return {"field": str(value), "type": "nominal"}

    @staticmethod
    def _default_type_for_channel(channel: str) -> str:
        if channel in {"x", "y", "x2", "y2", "size", "theta", "radius"}:
            return "quantitative"
        return "nominal"

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result
