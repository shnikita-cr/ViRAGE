from __future__ import annotations

import json
import textwrap
from pathlib import Path

from src.domain.models import (
    CodegenResult,
    DataPreparationResult,
    DataProfile,
    PlanningResult,
    QueryUnderstandingResult,
    VisualizationPlan,
    VisRAGResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_text
from src.services.base import BaseService

_WRAPPER_TEMPLATE = r'''
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PLAN_JSON = {plan_json!r}
QUERY_JSON = {query_json!r}
PLANNING_JSON = {planning_json!r}
PROFILE_JSON = {profile_json!r}
RETRIEVED_JSON = {retrieved_json!r}


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_temporal(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _aggregate_name(name: str | None) -> str:
    mapping = {{"mean": "mean", "sum": "sum", "count": "count", "median": "median", "max": "max", "min": "min"}}
    return mapping.get((name or "").lower(), "mean")


def _binding(plan: dict, channel: str) -> dict:
    for item in plan.get("field_bindings", []):
        if item.get("channel") == channel:
            return item
    return {{}}


def _axis_title(plan: dict, channel: str) -> str | None:
    for item in plan.get("axes", []):
        if item.get("channel") == channel:
            value = item.get("title")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _aggregate_from_plan(plan: dict) -> str:
    y_binding = _binding(plan, "y")
    if y_binding.get("aggregate"):
        return _aggregate_name(y_binding.get("aggregate"))
    for transform in plan.get("transforms", []):
        if transform.get("aggregate"):
            return _aggregate_name(transform.get("aggregate"))
    return "mean"


def _apply_aggregate(frame: pd.DataFrame, group_cols: list[str], value_col: str, agg: str) -> pd.DataFrame:
    if not group_cols:
        return frame
    return frame.groupby(group_cols, dropna=False)[value_col].agg(_aggregate_name(agg)).reset_index()


def main(output_dir: str = {output_dir!r}) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(r"{data_path}")
    plan = json.loads(PLAN_JSON)
    query_info = json.loads(QUERY_JSON)
    planning_info = json.loads(PLANNING_JSON)
    profile_info = json.loads(PROFILE_JSON)
    retrieved_examples = json.loads(RETRIEVED_JSON)

    chart_type = plan.get("chart_family", {chart_type!r})
    x_binding = _binding(plan, "x")
    y_binding = _binding(plan, "y")
    color_binding = _binding(plan, "color")
    x_col = x_binding.get("field_name")
    y_col = y_binding.get("field_name")
    color_col = color_binding.get("field_name")
    x_role = x_binding.get("field_role", "nominal")
    y_role = y_binding.get("field_role", "quantitative")
    aggregate_op = _aggregate_from_plan(plan)
    title = plan.get("title") or query_info.get("intent") or "Generated chart"
    subtitle = plan.get("subtitle")
    goal = plan.get("goal") or query_info.get("intent") or "Generate visualization"
    x_title = _axis_title(plan, "x") or x_col
    y_title = _axis_title(plan, "y") or y_col

    fig, ax = plt.subplots(figsize=(8, 5), dpi={dpi})
    metrics = {{
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "chart_type": chart_type,
        "aggregate": aggregate_op,
    }}
    plot_built = False

{generated_logic}

    if not plot_built:
        raise RuntimeError("Generated plotting logic did not mark the plot as built.")

    ax.set_title(title + ("\n" + subtitle if subtitle else ""))
    if x_title:
        ax.set_xlabel(x_title)
    if y_title:
        ax.set_ylabel(y_title)
    if x_role == "temporal":
        fig.autofmt_xdate()

    plot_path = out / "plot.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    metadata = {{
        "chart_type": chart_type,
        "title": title,
        "subtitle": subtitle,
        "goal": goal,
        "x_column": x_col,
        "y_column": y_col,
        "group_column": color_col,
        "aggregate": aggregate_op,
        "field_bindings": plan.get("field_bindings", []),
        "transforms": plan.get("transforms", []),
        "renderer_hints": plan.get("renderer_hints", []),
        "retrieved_example_ids": [item.get("example_id") for item in retrieved_examples],
        "plot_path": plot_path.as_posix(),
    }}

    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "chart_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


class CodegenService(BaseService):
    def invoke(
        self,
        query_understanding: QueryUnderstandingResult,
        planning: PlanningResult,
        data_profile: DataProfile,
        prepared: DataPreparationResult,
        visrag: VisRAGResult,
        run_id: str,
        runtime: RuntimeContext,
    ) -> CodegenResult:
        if runtime.codegen_llm is None:
            raise RuntimeError("Code generation requires runtime.codegen_llm. No codegen model was provided.")
        if visrag.visualization_plan is None:
            raise RuntimeError("Code generation requires visrag.visualization_plan, but VisRAG returned none.")

        plan = visrag.visualization_plan
        prompt = self._build_prompt(
            query_understanding=query_understanding,
            planning=planning,
            data_profile=data_profile,
            prepared=prepared,
            visrag=visrag,
            plan=plan,
        )
        run_dir = runtime.ensure_run_dir(run_id)
        prompt_path: str | None = None
        raw_response_path: str | None = None
        logic_path: str | None = None
        if runtime.settings.codegen_store_trace_artifacts:
            prompt_path = (run_dir / "codegen_prompt.txt").as_posix()
            Path(prompt_path).write_text(prompt, encoding="utf-8")

        raw_response = invoke_text(runtime.codegen_llm, prompt)
        if runtime.settings.codegen_store_trace_artifacts:
            raw_response_path = (run_dir / "codegen_raw_response.txt").as_posix()
            Path(raw_response_path).write_text(raw_response, encoding="utf-8")

        generated_logic = self._extract_code(raw_response)
        self._validate_generated_logic(generated_logic)
        if runtime.settings.codegen_store_trace_artifacts:
            logic_path = (run_dir / "generated_plot_logic.py").as_posix()
            Path(logic_path).write_text(generated_logic, encoding="utf-8")

        code = self._compose_script(
            prepared=prepared,
            query_understanding=query_understanding,
            planning=planning,
            data_profile=data_profile,
            visrag=visrag,
            plan=plan,
            generated_logic=generated_logic,
            runtime=runtime,
            run_id=run_id,
        )
        return CodegenResult(
            chart_type=plan.chart_family,
            code=code,
            prompt_path=prompt_path,
            raw_response_path=raw_response_path,
            generated_logic_path=logic_path,
        )

    def _build_prompt(
        self,
        *,
        query_understanding: QueryUnderstandingResult,
        planning: PlanningResult,
        data_profile: DataProfile,
        prepared: DataPreparationResult,
        visrag: VisRAGResult,
        plan: VisualizationPlan,
    ) -> str:
        examples = [item.model_dump() for item in visrag.retrieved_examples[:2]]
        return (
            "You generate ONLY the plotting logic body for a Python visualization pipeline.\n"
            "Return only executable Python statements. Do not return markdown fences.\n"
            "Do not import anything. Do not define functions or classes. Do not read files. Do not save files.\n"
            "You are writing the body inside main() after df, fig, ax, metrics and plan variables already exist.\n"
            "Available variables: df, ax, pd, plt, chart_type, x_col, y_col, color_col, x_role, y_role, aggregate_op, title, subtitle, goal, metrics, plan, query_info, planning_info, profile_info, retrieved_examples, plot_built.\n"
            "Available helper functions: _safe_numeric(series), _coerce_temporal(series), _apply_aggregate(frame, group_cols, value_col, agg), _aggregate_name(name).\n"
            "Requirements:\n"
            "1. Draw the chart on ax.\n"
            "2. Update metrics with the key columns and summary values you used.\n"
            "3. Set plot_built = True at the end when the plot is successfully created.\n"
            "4. Use only fields present in the VisualizationPlan when possible.\n"
            "5. Prefer simple, reliable pandas + matplotlib operations.\n\n"
            f"User query understanding JSON:\n{query_understanding.model_dump_json(indent=2)}\n\n"
            f"Planning JSON:\n{planning.model_dump_json(indent=2)}\n\n"
            f"Data profile JSON:\n{data_profile.model_dump_json(indent=2)}\n\n"
            f"Prepared data summary JSON:\n{prepared.model_dump_json(indent=2)}\n\n"
            f"Visualization plan JSON:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Retrieved examples JSON:\n{json.dumps(examples, ensure_ascii=False, indent=2)}\n\n"
            "Now return only the plotting logic body."
        )

    def _extract_code(self, raw_response: str) -> str:
        text = raw_response.strip()
        if "```" not in text:
            return text
        blocks = text.split("```")
        for block in blocks:
            cleaned = block.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered.startswith("python"):
                return cleaned[6:].lstrip("\n").strip()
            if any(token in cleaned for token in ["ax.", "plot_built", "metrics.update"]):
                return cleaned
        return text.replace("```", "").strip()

    def _validate_generated_logic(self, generated_logic: str) -> None:
        if not generated_logic.strip():
            raise RuntimeError("Codegen LLM returned an empty plotting logic block.")
        forbidden = [
            "import ",
            "from ",
            "def ",
            "class ",
            "__name__",
            "read_csv(",
            "savefig(",
            "subprocess",
            "requests",
            "open(",
            "eval(",
            "exec(",
            "os.",
            "sys.",
        ]
        lowered = generated_logic.lower()
        for token in forbidden:
            if token in lowered:
                raise RuntimeError(f"Generated plotting logic contains a forbidden token: {token}")
        required_markers = ["plot_built = true", "metrics.update"]
        for token in required_markers:
            if token not in lowered:
                raise RuntimeError(f"Generated plotting logic must contain: {token}")
        if "ax." not in generated_logic and "plt." not in generated_logic:
            raise RuntimeError("Generated plotting logic must draw on matplotlib axes.")

    def _compose_script(
        self,
        *,
        prepared: DataPreparationResult,
        query_understanding: QueryUnderstandingResult,
        planning: PlanningResult,
        data_profile: DataProfile,
        visrag: VisRAGResult,
        plan: VisualizationPlan,
        generated_logic: str,
        runtime: RuntimeContext,
        run_id: str,
    ) -> str:
        return _WRAPPER_TEMPLATE.format(
            output_dir=(runtime.ensure_run_dir(run_id) / "execution").resolve().as_posix(),
            data_path=Path(prepared.output_path).resolve().as_posix(),
            chart_type=plan.chart_family,
            plan_json=plan.model_dump_json(indent=2),
            query_json=query_understanding.model_dump_json(indent=2),
            planning_json=planning.model_dump_json(indent=2),
            profile_json=data_profile.model_dump_json(indent=2),
            retrieved_json=json.dumps([item.model_dump() for item in visrag.retrieved_examples[:3]], ensure_ascii=False, indent=2),
            generated_logic=textwrap.indent(generated_logic.strip(), "    "),
            dpi=runtime.settings.default_figure_dpi,
        )
