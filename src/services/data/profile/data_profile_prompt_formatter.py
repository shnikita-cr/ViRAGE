from __future__ import annotations

from src.domain.models import DataColumnProfile, DataProfile
from src.llm.prompt_budget import estimate_tokens


class DataProfilePromptFormatter:
    """Compact renderers for the canonical DataProfile."""

    @staticmethod
    def for_query_analysis(
        profile: DataProfile,
        max_columns: int = 30,
        token_budget: int | None = None,
    ) -> str:
        return DataProfilePromptFormatter._format(
            profile,
            max_columns=max_columns,
            include_samples=True,
            token_budget=token_budget,
        )

    @staticmethod
    def for_chart_generation(
        profile: DataProfile | None,
        max_columns: int = 30,
        token_budget: int | None = None,
    ) -> str:
        if profile is None:
            return "Dataset profile: unavailable. Use only explicitly provided field constraints."
        return DataProfilePromptFormatter._format(
            profile,
            max_columns=max_columns,
            include_samples=True,
            token_budget=token_budget,
        )

    @staticmethod
    def for_rag(profile: DataProfile, max_columns: int = 20) -> str:
        return DataProfilePromptFormatter._format(profile, max_columns=max_columns, include_samples=False)

    @staticmethod
    def for_debug(profile: DataProfile, max_columns: int = 100) -> str:
        return DataProfilePromptFormatter._format(profile, max_columns=max_columns, include_samples=True)

    @staticmethod
    def _format(
        profile: DataProfile,
        *,
        max_columns: int,
        include_samples: bool,
        token_budget: int | None = None,
    ) -> str:
        lines = DataProfilePromptFormatter._header(profile)
        omitted_by_budget = 0
        for column in profile.columns[:max_columns]:
            candidate = DataProfilePromptFormatter._column_line(column, include_samples=include_samples)
            if token_budget is not None and estimate_tokens("\n".join([*lines, candidate])) > token_budget:
                omitted_by_budget += 1
                continue
            lines.append(candidate)
        omitted = max(0, len(profile.columns) - max_columns) + omitted_by_budget
        if omitted:
            lines.append(f"... {omitted} more columns omitted")
        DataProfilePromptFormatter._append_notes(lines, profile, token_budget=token_budget)
        return "\n".join(lines)

    @staticmethod
    def _header(profile: DataProfile) -> list[str]:
        return [
            f"Rows={profile.row_count}; columns={profile.col_count}; complexity={profile.data_complexity or 'unknown'}; status={profile.profile_status}",
            "Columns: index | name | safe | type | role | missing | unique | range | samples | flags",
        ]

    @staticmethod
    def _column_line(column: DataColumnProfile, *, include_samples: bool) -> str:
        safe = column.safe_name or column.name
        range_text = DataProfilePromptFormatter._range_text(column)
        samples = ""
        if include_samples and column.sample_values:
            samples = "; samples=" + repr(column.sample_values[:3])
        flags = ""
        if column.quality_flags:
            flags = "; flags=" + ",".join(column.quality_flags[:4])
        hints = ""
        if column.preparation_hints:
            hints = "; prep=" + ",".join(column.preparation_hints[:3])
        return (
            f"- name={column.name!r}; safe={safe!r}; type={column.dtype}; role={column.role}; "
            f"missing={column.missing_ratio:.3f}; unique={column.unique_count}; range={range_text}{samples}{flags}{hints}"
        )

    @staticmethod
    def _range_text(column: DataColumnProfile) -> str:
        if column.min_value is None and column.max_value is None:
            return "n/a"
        return f"{column.min_value}..{column.max_value}"

    @staticmethod
    def _append_notes(lines: list[str], profile: DataProfile, *, token_budget: int | None) -> None:
        for label, values in (
            ("Quality notes", profile.quality_notes[:3]),
            ("Complexity hints", profile.complexity_hints[:3]),
            ("Profile errors", [str(item) for item in profile.errors[:3]]),
        ):
            if not values:
                continue
            candidate = f"{label}: " + "; ".join(values)
            if token_budget is None or estimate_tokens("\n".join([*lines, candidate])) <= token_budget:
                lines.append(candidate)
