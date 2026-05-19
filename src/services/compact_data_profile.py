from __future__ import annotations

import re
from typing import Any

from src.application.settings import ViRAGESettings
from src.domain.models import DataPreparationResult, DataProfile, RequestAnalysisResult
from src.services.base import BaseService

_SERVICE_COLUMN_RE = re.compile(
    r"(^|_)(artifact|artifacts|error|errors|trace|prompt|response|raw|parsed|run|path|paths|json|log|logs|model_call|stage)(_|$)",
    re.IGNORECASE,
)


class CompactDataProfileService(BaseService):
    """Build a small LLM-facing schema summary from the full data profile.

    The full profile remains saved as an artifact. This compact profile is used in prompts so wide service
    tables do not dominate the context window with JSON/path columns and long quality notes.
    """

    def invoke(
        self,
        data_profile: DataProfile,
        prepared: DataPreparationResult | None = None,
        request_analysis: RequestAnalysisResult | None = None,
        *,
        settings: ViRAGESettings,
    ) -> dict[str, Any]:
        selected_original = list(request_analysis.selected_fields if request_analysis is not None else [])
        if prepared is not None:
            selected_safe = [prepared.column_name_map.get(field, field) for field in selected_original]
            safe_columns = list(prepared.safe_columns)
            reverse_map = dict(prepared.reverse_column_name_map)
            column_map = dict(prepared.column_name_map)
        else:
            column_map = dict(data_profile.column_name_map or {})
            reverse_map = {safe: original for original, safe in column_map.items()}
            safe_columns = [column_map.get(column.name, column.safe_name or column.name) for column in data_profile.columns]
            selected_safe = [column_map.get(field, field) for field in selected_original]

        by_original = {column.original_name or column.name: column for column in data_profile.columns}
        by_name = {column.name: column for column in data_profile.columns}
        by_safe = {
            column.safe_name or data_profile.column_name_map.get(column.name, column.name): column
            for column in data_profile.columns
        }

        preferred_safe = list(dict.fromkeys([
            *selected_safe,
            *[data_profile.column_name_map.get(name, name) for name in data_profile.likely_time_columns],
            *[data_profile.column_name_map.get(name, name) for name in data_profile.likely_categorical_columns],
            *[data_profile.column_name_map.get(name, name) for name in data_profile.likely_numeric_columns],
            *safe_columns,
        ]))

        max_columns = int(settings.spec_generation_max_profile_columns)
        included: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        for safe in preferred_safe:
            if safe not in safe_columns:
                continue
            original = reverse_map.get(safe, safe)
            column = by_original.get(original) or by_name.get(original) or by_safe.get(safe)
            if self._should_exclude(original, safe, column, selected_original):
                excluded.append({"original": original, "safe": safe, "reason": self._exclude_reason(original, safe, column)})
                continue
            if len(included) >= max_columns and original not in selected_original:
                excluded.append({"original": original, "safe": safe, "reason": "max_profile_columns_limit"})
                continue
            included.append(self._column_payload(original, safe, column, data_profile, settings))

        candidate_dimensions = [item["safe"] for item in included if item.get("role") in {"dimension", "temporal"}]
        candidate_measures = [item["safe"] for item in included if item.get("role") == "measure"]

        return {
            "profile_type": "compact_data_profile",
            "row_count": data_profile.row_count,
            "column_count": data_profile.col_count,
            "data_complexity": data_profile.data_complexity,
            "included_column_count": len(included),
            "excluded_column_count": len(excluded),
            "selected_original_fields": selected_original,
            "selected_safe_fields": selected_safe,
            "candidate_dimensions": candidate_dimensions[:max_columns],
            "candidate_measures": candidate_measures[:max_columns],
            "columns": included,
            "excluded_columns": excluded[: max_columns * 2],
            "quality_notes_top": list(data_profile.quality_notes[: int(settings.spec_generation_max_quality_notes)]),
            "schema_hints_top": list(data_profile.schema_hints[: int(settings.spec_generation_max_quality_notes)]),
            "column_mapping_original_to_safe": {
                item["original"]: item["safe"] for item in included if item.get("original") != item.get("safe")
            },
        }

    def invoke_from_profile(
        self,
        data_profile: DataProfile,
        *,
        settings: ViRAGESettings,
    ) -> dict[str, Any]:
        return self.invoke(data_profile, prepared=None, request_analysis=None, settings=settings)

    @staticmethod
    def _should_exclude(original: str, safe: str, column: Any, selected_original: list[str]) -> bool:
        if original in selected_original:
            return False
        name = f"{original} {safe}"
        if _SERVICE_COLUMN_RE.search(name):
            return True
        if column is not None and bool(getattr(column, "is_identifier", False)):
            return True
        return False

    @staticmethod
    def _exclude_reason(original: str, safe: str, column: Any) -> str:
        name = f"{original} {safe}"
        if _SERVICE_COLUMN_RE.search(name):
            return "service_json_or_path_column"
        if column is not None and bool(getattr(column, "is_identifier", False)):
            return "identifier_like_column"
        return "excluded"

    @staticmethod
    def _column_payload(
        original: str,
        safe: str,
        column: Any,
        data_profile: DataProfile,
        settings: ViRAGESettings,
    ) -> dict[str, Any]:
        role = data_profile.field_roles.get(original, data_profile.field_roles.get(safe, "unknown"))
        payload: dict[str, Any] = {
            "original": original,
            "safe": safe,
            "type": getattr(column, "dtype", "unknown") if column is not None else "unknown",
            "role": role,
        }
        if column is not None:
            payload.update({
                "missing_ratio": round(float(getattr(column, "missing_ratio", 0.0) or 0.0), 6),
                "unique_count": int(getattr(column, "unique_count", 0) or 0),
                "min": getattr(column, "min_value", None),
                "max": getattr(column, "max_value", None),
                "sample_values": list(getattr(column, "sample_values", []) or [])[: int(settings.spec_generation_max_sample_values)],
            })
        return payload
