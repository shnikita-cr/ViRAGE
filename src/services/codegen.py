from pathlib import Path

from src.domain.models import (
    CodegenResult,
    DataPreparationResult,
    DataProfile,
    QueryUnderstandingResult,
    VisualizationPlan,
    VisRAGResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService

_TEMPLATE = r'''
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _safe_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")


def _aggregate_name(name: str):
    mapping = {{"mean": "mean", "sum": "sum", "count": "count", "median": "median"}}
    return mapping.get((name or "").lower(), "mean")


def main(output_dir: str = "{output_dir}") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(r"{data_path}")
    chart_type = {chart_type!r}
    x_col = {x_col!r}
    y_col = {y_col!r}
    color_col = {color_col!r}
    aggregate_op = {aggregate_op!r}
    x_role = {x_role!r}
    title = {title!r}
    subtitle = {subtitle!r}
    x_title = {x_title!r}
    y_title = {y_title!r}

    fig, ax = plt.subplots(figsize=(8, 5), dpi={dpi})
    metrics = {{"row_count": int(len(df)), "column_count": int(len(df.columns)), "chart_type": chart_type}}
    agg = _aggregate_name(aggregate_op)

    if x_col and x_role == "temporal" and x_col in df.columns:
        df[x_col] = pd.to_datetime(df[x_col], errors="coerce")

    if chart_type == "line" and y_col:
        if x_col and x_col in df.columns:
            if color_col and color_col in df.columns:
                grouped = df[[x_col, y_col, color_col]].dropna(subset=[y_col]).groupby([x_col, color_col], dropna=False)[y_col].agg(agg).reset_index()
                for group_name, group_df in grouped.groupby(color_col):
                    ax.plot(group_df[x_col], group_df[y_col], marker="o", label=str(group_name))
                ax.legend(title=color_col)
                metrics.update({{"group_column": color_col, "group_count": int(grouped[color_col].nunique())}})
            else:
                grouped = df[[x_col, y_col]].dropna(subset=[y_col]).groupby(x_col, dropna=False)[y_col].agg(agg).reset_index()
                ax.plot(grouped[x_col], grouped[y_col], marker="o")
            metrics.update({{"x_column": x_col, "y_column": y_col, "aggregate": agg, "y_min": float(grouped[y_col].min()), "y_max": float(grouped[y_col].max()), "y_mean": float(grouped[y_col].mean())}})
        else:
            y = _safe_numeric(df[y_col])
            ax.plot(range(len(y)), y, marker="o")
            metrics.update({{"x_column": "index", "y_column": y_col, "aggregate": "none", "y_min": float(y.min()), "y_max": float(y.max()), "y_mean": float(y.mean())}})
    elif chart_type == "scatter" and y_col:
        x_series = _safe_numeric(df[x_col]) if x_col and x_col in df.columns else pd.Series(range(len(df)))
        y_series = _safe_numeric(df[y_col])
        if color_col and color_col in df.columns:
            for group_name, group_df in df.groupby(color_col, dropna=False):
                ax.scatter(_safe_numeric(group_df[x_col]) if x_col else range(len(group_df)), _safe_numeric(group_df[y_col]), label=str(group_name))
            ax.legend(title=color_col)
        else:
            ax.scatter(x_series, y_series)
        metrics.update({{"x_column": x_col or "index", "y_column": y_col, "x_min": float(x_series.min()), "x_max": float(x_series.max()), "y_min": float(y_series.min()), "y_max": float(y_series.max())}})
    elif chart_type == "histogram" and x_col:
        x_series = _safe_numeric(df[x_col])
        ax.hist(x_series.dropna(), bins=10)
        metrics.update({{"x_column": x_col, "y_column": "count", "x_min": float(x_series.min()), "x_max": float(x_series.max())}})
    elif chart_type == "boxplot" and y_col:
        y = _safe_numeric(df[y_col])
        if x_col and x_col in df.columns:
            grouped = [grp[y_col].dropna().tolist() for _, grp in df[[x_col, y_col]].groupby(x_col)]
            labels = [str(k) for k, _ in df[[x_col, y_col]].groupby(x_col)]
            ax.boxplot(grouped, labels=labels)
            metrics.update({{"group_column": x_col, "y_column": y_col, "groups": labels}})
        else:
            ax.boxplot(y.dropna())
            metrics.update({{"y_column": y_col}})
    elif y_col:
        if x_col and x_col in df.columns:
            grouped = df[[x_col, y_col]].dropna(subset=[y_col]).groupby(x_col, dropna=False)[y_col].agg(agg).reset_index()
            ax.bar(grouped[x_col].astype(str), grouped[y_col])
            metrics.update({{"x_column": x_col, "y_column": y_col, "aggregate": agg, "group_count": int(len(grouped)), "y_mean": float(grouped[y_col].mean())}})
        else:
            y = _safe_numeric(df[y_col])
            ax.bar(range(len(y)), y)
            metrics.update({{"x_column": "index", "y_column": y_col, "aggregate": "none", "y_mean": float(y.mean())}})

    ax.set_title(title + ("\n" + subtitle if subtitle else ""))
    if x_col or x_title:
        ax.set_xlabel(x_title or x_col)
    if y_col or y_title:
        ax.set_ylabel(y_title or y_col)
    fig.autofmt_xdate()
    plot_path = out / "plot.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    metadata = {{
        "chart_type": chart_type,
        "title": title,
        "subtitle": subtitle,
        "x_column": x_col,
        "y_column": y_col,
        "group_column": color_col,
        "aggregate": aggregate_op,
        "plot_path": plot_path.as_posix(),
        "visualization_goal": {goal!r},
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
        data_profile: DataProfile,
        prepared: DataPreparationResult,
        visrag: VisRAGResult,
        run_id: str,
        runtime: RuntimeContext,
    ) -> CodegenResult:
        plan = visrag.visualization_plan or self._fallback_plan(query_understanding, data_profile, visrag)
        chart_type = plan.chart_family
        x_col = self._field_for(plan, "x")
        y_col = self._field_for(plan, "y")
        color_col = self._field_for(plan, "color")
        aggregate_op = self._aggregate_for(plan, preferred_channel="y")
        code = _TEMPLATE.format(
            output_dir=(runtime.ensure_run_dir(run_id) / "execution").resolve().as_posix(),
            data_path=Path(prepared.output_path).resolve().as_posix(),
            chart_type=chart_type,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            aggregate_op=aggregate_op,
            x_role=self._role_for(plan, "x"),
            title=plan.title,
            subtitle=plan.subtitle,
            x_title=self._axis_title(plan, "x"),
            y_title=self._axis_title(plan, "y"),
            goal=plan.goal,
            dpi=runtime.settings.default_figure_dpi,
        )
        return CodegenResult(chart_type=chart_type, code=code)

    def _fallback_plan(self, query_understanding: QueryUnderstandingResult, profile: DataProfile, visrag: VisRAGResult) -> VisualizationPlan:
        chart_type = visrag.recommendations[0].chart_family if visrag.recommendations else "bar"
        x_col, y_col = self._pick_fields(chart_type, profile)
        field_bindings = []
        if x_col:
            field_bindings.append({"channel": "x", "field_name": x_col, "field_role": "temporal" if x_col in profile.likely_time_columns else "nominal"})
        if y_col:
            field_bindings.append({"channel": "y", "field_name": y_col, "field_role": "quantitative"})
        return VisualizationPlan(chart_family=chart_type, visual_task="fallback", goal=query_understanding.intent, title=query_understanding.intent, field_bindings=field_bindings)

    @staticmethod
    def _field_for(plan: VisualizationPlan, channel: str) -> str | None:
        for binding in plan.field_bindings:
            if binding.channel == channel:
                return binding.field_name
        return None

    @staticmethod
    def _role_for(plan: VisualizationPlan, channel: str) -> str:
        for binding in plan.field_bindings:
            if binding.channel == channel:
                return binding.field_role
        return "nominal"

    @staticmethod
    def _aggregate_for(plan: VisualizationPlan, preferred_channel: str) -> str:
        for binding in plan.field_bindings:
            if binding.channel == preferred_channel and binding.aggregate:
                return binding.aggregate
        for transform in plan.transforms:
            if transform.aggregate:
                return transform.aggregate
        return "mean"

    @staticmethod
    def _axis_title(plan: VisualizationPlan, channel: str) -> str | None:
        for axis in plan.axes:
            if axis.channel == channel:
                return axis.title
        return None

    def _pick_fields(self, chart_type: str, profile: DataProfile) -> tuple[str | None, str | None]:
        numeric = profile.likely_numeric_columns
        categorical = profile.likely_categorical_columns
        time_like = profile.likely_time_columns
        if chart_type == "line":
            return (time_like[0] if time_like else None, numeric[0] if numeric else None)
        if chart_type == "scatter":
            if len(numeric) >= 2:
                return numeric[0], numeric[1]
            return None, numeric[0] if numeric else None
        if chart_type == "histogram":
            return numeric[0] if numeric else None, None
        if chart_type == "boxplot":
            return categorical[0] if categorical else None, numeric[0] if numeric else None
        return (categorical[0] if categorical else time_like[0] if time_like else None, numeric[0] if numeric else None)
