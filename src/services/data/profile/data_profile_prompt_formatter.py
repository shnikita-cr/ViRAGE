from __future__ import annotations

from src.domain.models import DataProfile


class DataProfilePromptFormatter:
    """Text renderers for the single canonical DataProfile.

    This class does not create another profile object. It only prepares concise
    prompt text for services that need to pass schema context to an LLM.
    """

    @staticmethod
    def for_query_analysis(profile: DataProfile, max_columns: int = 30) -> str:
        return DataProfilePromptFormatter._format(profile, max_columns=max_columns, include_samples=True)

    @staticmethod
    def for_chart_generation(profile: DataProfile | None, max_columns: int = 30) -> str:
        if profile is None:
            return "Dataset profile: unavailable. Use only explicitly provided field constraints."
        return DataProfilePromptFormatter._format(profile, max_columns=max_columns, include_samples=True)

    @staticmethod
    def for_rag(profile: DataProfile, max_columns: int = 20) -> str:
        return DataProfilePromptFormatter._format(profile, max_columns=max_columns, include_samples=False)

    @staticmethod
    def for_debug(profile: DataProfile, max_columns: int = 100) -> str:
        return DataProfilePromptFormatter._format(profile, max_columns=max_columns, include_samples=True)

    @staticmethod
    def _format(profile: DataProfile, *, max_columns: int, include_samples: bool) -> str:
        lines = [
            f"Rows={profile.row_count}; columns={profile.col_count}; complexity={profile.data_complexity or 'unknown'}; status={profile.profile_status}",
            "Columns:",
        ]
        for column in profile.columns[:max_columns]:
            safe = column.safe_name or column.name
            parts = [
                f"- {column.name}",
                f"safe={safe}",
                f"type={column.dtype}",
                f"role={column.role}",
                f"missing={column.missing_ratio:.3f}",
                f"unique={column.unique_count}",
            ]
            if column.min_value is not None:
                parts.append(f"min={column.min_value}")
            if column.max_value is not None:
                parts.append(f"max={column.max_value}")
            if column.quality_flags:
                parts.append("flags=" + ",".join(column.quality_flags[:8]))
            if column.preparation_hints:
                parts.append("prep=" + ",".join(column.preparation_hints[:6]))
            if include_samples and column.sample_values:
                parts.append("samples=" + repr(column.sample_values[:5]))
            lines.append(" | ".join(parts))
        if len(profile.columns) > max_columns:
            lines.append(f"... {len(profile.columns) - max_columns} more columns omitted")
        if profile.quality_notes:
            lines.append("Quality notes: " + "; ".join(profile.quality_notes[:5]))
        if profile.complexity_hints:
            lines.append("Complexity hints: " + "; ".join(profile.complexity_hints[:5]))
        if profile.errors:
            lines.append("Profile errors: " + "; ".join(str(item) for item in profile.errors[:5]))
        return "\n".join(lines)
