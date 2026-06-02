from __future__ import annotations

from src.orchestrator.models import AnalysisPlan, OrchestratorReport


class FinalSummaryBuilder:
    def build(self, *, plan: AnalysisPlan, report: OrchestratorReport) -> str:
        lines: list[str] = [
            "# ViRAGE orchestrator summary",
            "",
            f"User query: {plan.user_query}",
            f"Input type: {plan.input_type}",
            f"Data path: {plan.data_path}",
            f"Original input path: {plan.original_input_path or plan.data_path}",
            f"Preprocessing report: {plan.preprocessing_report_path or 'not applicable'}",
            f"Executed: {str(report.executed).lower()}",
            "",
            "## Planned analytical subtasks",
            "",
        ]
        for item in plan.subtasks:
            lines.extend([
                f"### {item.id}",
                "",
                f"- Task type: `{item.task_type}`",
                f"- Purpose: {item.purpose}",
                f"- Query: {item.query}",
                f"- Required fields: {', '.join(item.required_fields) if item.required_fields else 'none'}",
                f"- Rationale: {item.rationale or 'not specified'}",
                "",
            ])
        if plan.skipped_candidates:
            lines.extend(["## Skipped candidates", ""])
            for item in plan.skipped_candidates:
                lines.append(f"- `{item.task_type}`: {item.reason}")
            lines.append("")
        if report.subruns:
            lines.extend(["## Subrun results", ""])
            for item in report.subruns:
                lines.append(f"- `{item.subtask_id}`: {item.status}; run id: `{item.run_id}`")
            lines.append("")
        lines.extend([
            "## Notes",
            "",
            "The orchestrator selects analytical tasks only. It does not replace task-specific RAG guidance or Vega-Lite spec generation.",
        ])
        return "\n".join(lines).strip() + "\n"
