from src.domain.models import CodegenResult, DataPreparationResult, DataProfile, QueryUnderstandingResult, VisRAGResult
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


def main(output_dir: str = "{output_dir}") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(r"{data_path}")
    chart_type = {chart_type!r}
    x_col = {x_col!r}
    y_col = {y_col!r}

    fig, ax = plt.subplots(figsize=(8, 5), dpi={dpi})
    title = {query!r}
    metrics = {{"row_count": int(len(df)), "column_count": int(len(df.columns)), "chart_type": chart_type}}

    if chart_type == "line" and y_col:
        x = pd.to_datetime(df[x_col]) if x_col else pd.RangeIndex(len(df))
        y = _safe_numeric(df[y_col])
        ax.plot(x, y, marker="o")
        metrics.update({{"x_column": x_col or "index", "y_column": y_col, "y_min": float(y.min()), "y_max": float(y.max()), "y_mean": float(y.mean())}})
    elif chart_type == "scatter" and y_col:
        if x_col:
            x = _safe_numeric(df[x_col])
            y = _safe_numeric(df[y_col])
            ax.scatter(x, y)
            metrics.update({{"x_column": x_col, "y_column": y_col, "x_min": float(x.min()), "x_max": float(x.max()), "y_min": float(y.min()), "y_max": float(y.max())}})
        else:
            y = _safe_numeric(df[y_col])
            ax.scatter(range(len(y)), y)
            metrics.update({{"x_column": "index", "y_column": y_col, "y_min": float(y.min()), "y_max": float(y.max()), "y_mean": float(y.mean())}})
    elif chart_type == "histogram" and y_col:
        y = _safe_numeric(df[y_col])
        ax.hist(y.dropna(), bins=10)
        metrics.update({{"y_column": y_col, "y_min": float(y.min()), "y_max": float(y.max()), "y_mean": float(y.mean())}})
    elif chart_type == "boxplot" and y_col:
        y = _safe_numeric(df[y_col])
        if x_col and x_col in df.columns:
            groups = [grp[y_col].dropna().tolist() for _, grp in df[[x_col, y_col]].groupby(x_col)]
            labels = [str(k) for k, _ in df[[x_col, y_col]].groupby(x_col)]
            ax.boxplot(groups, labels=labels)
            metrics.update({{"group_column": x_col, "y_column": y_col, "groups": labels}})
        else:
            ax.boxplot(y.dropna())
            metrics.update({{"y_column": y_col}})
    elif y_col:
        if x_col and x_col in df.columns:
            grouped = df[[x_col, y_col]].groupby(x_col, dropna=False).mean(numeric_only=True).reset_index()
            ax.bar(grouped[x_col].astype(str), grouped[y_col])
            metrics.update({{"x_column": x_col, "y_column": y_col, "group_count": int(len(grouped)), "y_mean": float(grouped[y_col].mean())}})
        else:
            y = _safe_numeric(df[y_col])
            ax.bar(range(len(y)), y)
            metrics.update({{"x_column": "index", "y_column": y_col, "y_mean": float(y.mean())}})

    ax.set_title(title)
    if x_col:
        ax.set_xlabel(x_col)
    if y_col:
        ax.set_ylabel(y_col)
    fig.autofmt_xdate()
    plot_path = out / "plot.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    metadata = {{
        "chart_type": chart_type,
        "title": title,
        "x_column": x_col,
        "y_column": y_col,
        "plot_path": plot_path.as_posix(),
    }}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "chart_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


class CodegenService(BaseService):
    def invoke(self, query_understanding: QueryUnderstandingResult, data_profile: DataProfile,
               prepared: DataPreparationResult, visrag: VisRAGResult, run_id: str,
               runtime: RuntimeContext) -> CodegenResult:
        chart_type = visrag.recommendations[0].chart_family if visrag.recommendations else "bar"
        x_col, y_col = self._pick_fields(chart_type, data_profile)
        code = _TEMPLATE.format(
            output_dir=(runtime.ensure_run_dir(run_id) / "execution").as_posix(),
            data_path=prepared.output_path,
            chart_type=chart_type,
            x_col=x_col,
            y_col=y_col,
            query=query_understanding.intent,
            dpi=runtime.settings.default_figure_dpi,
        )
        return CodegenResult(chart_type=chart_type, code=code)

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
            return None, numeric[0] if numeric else None
        if chart_type == "boxplot":
            return categorical[0] if categorical else None, numeric[0] if numeric else None
        return (categorical[0] if categorical else time_like[0] if time_like else None, numeric[0] if numeric else None)
