from __future__ import annotations

import re
from typing import Any

from src.application.config.settings import ViRAGESettings
from src.domain.models import DataPreparationResult, DataProfile, QueryRequestAnalysisResult
from src.services.base import BaseService

_SERVICE_COLUMN_RE = re.compile(
    r"(^|_)(artifact|artifacts|error|errors|trace|prompt|response|raw|parsed|run|path|paths|json|log|logs|model_call|stage)(_|$)",
    re.IGNORECASE,
)


class CompactDataProfileService(BaseService):
    """Build a compact LLM-facing schema summary from the full DataProfile."""

    def invoke(
            self,
            data_profile: DataProfile,
            prepared: DataPreparationResult | None = None,
            request_analysis: QueryRequestAnalysisResult | None = None,
            *,
            settings: ViRAGESettings,
    ) -> dict[str, Any]:
        selected_original = list(request_analysis.selected_fields if request_analysis is not None else [])
        column_map = prepared.column_name_map if prepared is not None else data_profile.original_to_safe_map()
        reverse_map = prepared.reverse_column_name_map if prepared is not None else data_profile.safe_to_original_map()
        safe_columns = list(prepared.safe_columns if prepared is not None else column_map.values())
        selected_safe = [column_map.get(field, field) for field in selected_original]
        column_by_original = {column.name: column for column in data_profile.columns}
        preferred_safe = self._preferred_safe_columns(data_profile, safe_columns, selected_safe, column_map)
        included, excluded = self._collect_columns(
            preferred_safe=preferred_safe,
            safe_columns=set(safe_columns),
            reverse_map=dict(reverse_map),
            column_by_original=column_by_original,
            selected_original=selected_original,
            data_profile=data_profile,
            settings=settings,
        )
        return self._payload(data_profile, included, excluded, selected_original, selected_safe, settings)

    def invoke_from_profile(
            self,
            data_profile: DataProfile,
            *,
            settings: ViRAGESettings,
    ) -> dict[str, Any]:
        return self.invoke(data_profile, prepared=None, request_analysis=None, settings=settings)

    @staticmethod
    def _preferred_safe_columns(
            data_profile: DataProfile,
            safe_columns: list[str],
            selected_safe: list[str],
            column_map: dict[str, str],
    ) -> list[str]:
        role_order = ["temporal", "dimension", "measure"]
        role_fields = [
            column_map.get(column.name, column.safe_name or column.name)
            for role in role_order
            for column in data_profile.columns_by_role(role)
        ]
        return list(dict.fromkeys([*selected_safe, *role_fields, *safe_columns]))

    @classmethod
    def _collect_columns(
            cls,
            *,
            preferred_safe: list[str],
            safe_columns: set[str],
            reverse_map: dict[str, str],
            column_by_original: dict[str, Any],
            selected_original: list[str],
            data_profile: DataProfile,
            settings: ViRAGESettings,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        max_columns = int(settings.spec_generation_max_profile_columns)
        included: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        for safe in preferred_safe:
            if safe not in safe_columns:
                continue
            original = reverse_map.get(safe, safe)
            column = column_by_original.get(original)
            if cls._should_exclude(original, safe, column, selected_original):
                excluded.append({"original": original, "safe": safe, "reason": cls._exclude_reason(original, safe, column)})
                continue
            if len(included) >= max_columns and original not in selected_original:
                excluded.append({"original": original, "safe": safe, "reason": "max_profile_columns_limit"})
                continue
            included.append(cls._column_payload(original, safe, column, data_profile, settings))
        return included, excluded

    @staticmethod
    def _payload(
            data_profile: DataProfile,
            included: list[dict[str, Any]],
            excluded: list[dict[str, str]],
            selected_original: list[str],
            selected_safe: list[str],
            settings: ViRAGESettings,
    ) -> dict[str, Any]:
        max_columns = int(settings.spec_generation_max_profile_columns)
        return {
            "profile_type": "compact_data_profile",
            "row_count": data_profile.row_count,
            "column_count": data_profile.col_count,
            "data_complexity": data_profile.data_complexity,
            "included_column_count": len(included),
            "excluded_column_count": len(excluded),
            "selected_original_fields": selected_original,
            "selected_safe_fields": selected_safe,
            "candidate_dimensions": [item["safe"] for item in included if item.get("role") in {"dimension", "temporal"}][:max_columns],
            "candidate_measures": [item["safe"] for item in included if item.get("role") == "measure"][:max_columns],
            "columns": included,
            "excluded_columns": excluded[: max_columns * 2],
            "quality_notes_top": list(data_profile.quality_notes[: int(settings.spec_generation_max_quality_notes)]),
            "schema_hints_top": list(data_profile.complexity_hints[: int(settings.spec_generation_max_quality_notes)]),
            "column_mapping_original_to_safe": {item["original"]: item["safe"] for item in included if item.get("original") != item.get("safe")},
        }

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
        role = getattr(column, "role", "unknown") if column is not None else "unknown"
        payload: dict[str, Any] = {"original": original, "safe": safe, "type": getattr(column, "dtype", "unknown") if column is not None else "unknown", "role": role}
        if column is not None:
            payload.update({
                "missing_ratio": round(float(getattr(column, "missing_ratio", 0.0) or 0.0), 6),
                "unique_count": int(getattr(column, "unique_count", 0) or 0),
                "min": getattr(column, "min_value", None),
                "max": getattr(column, "max_value", None),
                "sample_values": list(getattr(column, "sample_values", []) or [])[: int(settings.spec_generation_max_sample_values)],
                "flags": list(getattr(column, "quality_flags", []) or []),
                "recommended_preparation": list(getattr(column, "preparation_hints", []) or []),
            })
        return payload
