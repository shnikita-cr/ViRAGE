from __future__ import annotations

from typing import Any

from src.domain.models import DataProfile, QueryRequestAnalysisResult
from src.visrag_core.task_context import task_context_to_query_text


def build_visrag_query(
        query_analysis: QueryRequestAnalysisResult,
        data_profile: DataProfile,
        task_context: dict[str, Any] | None = None,
) -> str:
    fields = " ".join(query_analysis.selected_fields)
    bindings = " ".join(
        f"{slot} {binding.field} {binding.role}" for slot, binding in query_analysis.field_bindings.items()
    )
    variants = " ".join(str(getattr(variant, "text", variant)) for variant in query_analysis.query_variants[:6])
    columns = " ".join(column.name for column in data_profile.columns[:30])
    task_text = task_context_to_query_text(task_context)
    return " ".join([
        task_text,
        query_analysis.normalized_query,
        query_analysis.analysis_task,
        fields,
        bindings,
        variants,
        columns,
    ]).strip()
