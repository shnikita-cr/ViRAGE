from __future__ import annotations

from typing import Any

_TASK_CONTEXT_KEYS = {
    "id",
    "task_type",
    "query",
    "purpose",
    "required_fields",
    "optional_fields",
    "constraints",
    "rationale",
}


def task_context_from_user_context(user_context: dict[str, Any] | None) -> dict[str, Any]:
    """Extract orchestrator-selected analytical subtask for task-specific RAG.

    The function intentionally accepts plain dictionaries to avoid coupling the
    core RAG package to the orchestrator package. Runtime RAG must treat this
    context as a fixed task contract: it can use the context to retrieve and
    format guidance, but must not replace the selected analytical task.
    """
    if not isinstance(user_context, dict):
        return {}
    raw = user_context.get("analysis_subtask")
    if not isinstance(raw, dict):
        return {}
    result: dict[str, Any] = {}
    for key in _TASK_CONTEXT_KEYS:
        value = raw.get(key)
        if value in (None, "", [], {}):
            continue
        result[key] = value
    constraints = result.get("constraints")
    if isinstance(constraints, dict):
        constraints.setdefault("output_target", "scientific_figure")
        constraints.setdefault("task_is_fixed", True)
    elif result:
        result["constraints"] = {"output_target": "scientific_figure", "task_is_fixed": True}
    return result


def task_context_to_query_text(task_context: dict[str, Any] | None) -> str:
    if not task_context:
        return ""
    fields = _join_items(task_context.get("required_fields"))
    optional = _join_items(task_context.get("optional_fields"))
    constraints = task_context.get("constraints") if isinstance(task_context.get("constraints"), dict) else {}
    constraint_text = " ".join(f"{key} {value}" for key, value in sorted(constraints.items()))
    return " ".join(
        part
        for part in [
            "selected analytical subtask",
            str(task_context.get("task_type") or ""),
            str(task_context.get("purpose") or ""),
            str(task_context.get("query") or ""),
            fields,
            optional,
            constraint_text,
            "scientific figure publication ready static chart",
        ]
        if part.strip()
    ).strip()


def task_context_prompt_block(task_context: dict[str, Any] | None) -> str:
    if not task_context:
        return ""
    lines = [
        "Selected analytical task contract:",
        f"- task_type: {task_context.get('task_type', '')}",
        f"- purpose: {task_context.get('purpose', '')}",
        f"- required_fields: {_join_items(task_context.get('required_fields')) or 'none'}",
        f"- optional_fields: {_join_items(task_context.get('optional_fields')) or 'none'}",
        "- output_target: scientific_figure",
        "- The analytical task is already selected by the orchestrator.",
        "- Use RAG only as rules and constraints for this selected task.",
        "- Do not replace this task with another analytical task.",
    ]
    constraints = task_context.get("constraints") if isinstance(task_context.get("constraints"), dict) else {}
    if constraints:
        lines.append("- constraints:")
        lines.extend(f"  - {key}: {value}" for key, value in sorted(constraints.items()))
    rationale = str(task_context.get("rationale") or "").strip()
    if rationale:
        lines.append(f"- rationale: {rationale}")
    return "\n".join(lines)


def _join_items(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()
