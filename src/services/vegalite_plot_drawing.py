from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.domain.models import PlotImageArtifact, PlotRenderingResult, SpecValidationResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class VegaLitePlotDrawingService(BaseService):
    def invoke(self, spec_validation: SpecValidationResult, run_id: str,
               runtime: RuntimeContext) -> PlotRenderingResult:
        if not spec_validation.is_valid:
            raise RuntimeError("Cannot draw a Vega-Lite plot from an invalid specification.")
        spec = spec_validation.validated_spec
        data_url = spec["data"]["url"]
        df = pd.read_csv(data_url)
        df = self._apply_transforms(df, spec.get("transform", []))
        run_dir = runtime.ensure_run_dir(run_id)
        image_path = run_dir / "plot.png"

        width = int(spec.get("width", 8 * runtime.settings.default_figure_dpi))
        height = int(spec.get("height", 5 * runtime.settings.default_figure_dpi))
        fig, ax = plt.subplots(figsize=(
        max(width, 320) / runtime.settings.default_figure_dpi, max(height, 240) / runtime.settings.default_figure_dpi),
                               dpi=runtime.settings.default_figure_dpi)
        mark = spec.get("mark")
        mark_type = mark.get("type") if isinstance(mark, dict) else mark
        encoding = spec.get("encoding", {})
        x = encoding.get("x", {})
        y = encoding.get("y", {})
        color = encoding.get("color", {})
        x_field = x.get("field")
        y_field = y.get("field")
        color_field = color.get("field")
        aggregate = y.get("aggregate")
        marks_count = 0
        notes: list[str] = []

        if x.get("type") == "temporal" and x_field in df.columns:
            df[x_field] = pd.to_datetime(df[x_field], errors="coerce")
        if y_field in df.columns:
            df[y_field] = pd.to_numeric(df[y_field], errors="coerce")
        if color_field in df.columns and color.get("type") == "quantitative":
            df[color_field] = pd.to_numeric(df[color_field], errors="coerce")

        if mark_type in {"line", "area"}:
            plot_df = df[[col for col in [x_field, y_field, color_field] if col in df.columns]].dropna(
                subset=[c for c in [x_field, y_field] if c]).copy()
            if aggregate and x_field:
                grouping = [x_field] + ([color_field] if color_field and color_field in plot_df.columns else [])
                plot_df = plot_df.groupby(grouping, dropna=False)[y_field].agg(aggregate).reset_index()
            plot_df = plot_df.sort_values([c for c in [color_field, x_field] if c])
            if color_field and color_field in plot_df.columns:
                for label, frame in plot_df.groupby(color_field, dropna=False):
                    ax.plot(frame[x_field], frame[y_field], marker="o", label=str(label))
                    if mark_type == "area":
                        ax.fill_between(frame[x_field], frame[y_field], alpha=0.2)
                    marks_count += int(len(frame))
                ax.legend()
            else:
                ax.plot(plot_df[x_field], plot_df[y_field], marker="o")
                if mark_type == "area":
                    ax.fill_between(plot_df[x_field], plot_df[y_field], alpha=0.2)
                marks_count = int(len(plot_df))
        elif mark_type == "bar":
            plot_df = df[[col for col in [x_field, y_field, color_field] if col in df.columns]].dropna(
                subset=[c for c in [x_field, y_field] if c]).copy()
            if aggregate and x_field:
                grouping = [x_field] + ([color_field] if color_field and color_field in plot_df.columns else [])
                plot_df = plot_df.groupby(grouping, dropna=False)[y_field].agg(aggregate).reset_index()
            if color_field and color_field in plot_df.columns:
                pivot = plot_df.pivot(index=x_field, columns=color_field, values=y_field).fillna(0)
                pivot.plot(kind="bar", ax=ax)
                marks_count = int(pivot.size)
            else:
                ax.bar(plot_df[x_field].astype(str), plot_df[y_field])
                marks_count = int(len(plot_df))
        elif mark_type in {"point", "circle", "tick"}:
            plot_df = df[[col for col in [x_field, y_field, color_field] if col in df.columns]].dropna(
                subset=[c for c in [x_field, y_field] if c]).copy()
            if color_field and color_field in plot_df.columns:
                for label, frame in plot_df.groupby(color_field, dropna=False):
                    ax.scatter(frame[x_field], frame[y_field], label=str(label))
                    marks_count += int(len(frame))
                ax.legend()
            else:
                ax.scatter(plot_df[x_field], plot_df[y_field])
                marks_count = int(len(plot_df))
        elif mark_type == "histogram":
            series = pd.to_numeric(df[x_field], errors="coerce").dropna()
            ax.hist(series, bins=10)
            marks_count = int(len(series))
        elif mark_type == "boxplot":
            if color_field and color_field in df.columns:
                groups = []
                labels = []
                for label, frame in df.groupby(color_field, dropna=False):
                    series = pd.to_numeric(frame[y_field], errors="coerce").dropna()
                    if not series.empty:
                        groups.append(series)
                        labels.append(str(label))
                if groups:
                    ax.boxplot(groups, tick_labels=labels)
                    marks_count = sum(len(group) for group in groups)
            else:
                series = pd.to_numeric(df[y_field], errors="coerce").dropna()
                ax.boxplot(series)
                marks_count = int(len(series))
        else:
            raise RuntimeError(f"Unsupported rendered mark type: {mark_type}")

        ax.set_title(spec.get("title") or "Generated visualization")
        if x_field:
            ax.set_xlabel(x_field)
        if y_field:
            ax.set_ylabel(y_field)
        if x.get("type") == "temporal":
            fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(image_path)
        pixel_width, pixel_height = map(int, fig.get_size_inches() * fig.dpi)
        plt.close(fig)

        scenegraph = {
            "mark_type": mark_type,
            "marks_count": marks_count,
            "axes": [axis for axis in [x_field, y_field] if axis],
            "has_legend": bool(color_field),
            "notes": notes,
        }
        return PlotRenderingResult(
            plot_image=PlotImageArtifact(image_path=image_path.as_posix(), width=pixel_width, height=pixel_height),
            rendered_scenegraph=scenegraph,
            render_notes=notes,
        )

    @staticmethod
    def _apply_transforms(df: pd.DataFrame, transforms: list[dict[str, Any]]) -> pd.DataFrame:
        result = df.copy()
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            kind = transform.get("kind")
            if kind == "filter":
                expression = transform.get("filter") or transform.get("expression")
                if isinstance(expression, str) and expression.strip():
                    try:
                        result = result.query(expression)
                    except Exception:
                        continue
            elif kind == "calculate":
                as_name = transform.get("as") or transform.get("field_name")
                expression = transform.get("calculate") or transform.get("expression")
                if isinstance(as_name, str) and isinstance(expression, str) and expression.strip():
                    try:
                        result[as_name] = result.eval(expression)
                    except Exception:
                        continue
            elif kind == "bin":
                field = transform.get("field") or transform.get("field_name")
                if isinstance(field, str) and field in result.columns:
                    numeric = pd.to_numeric(result[field], errors="coerce")
                    if numeric.notna().sum() >= 2:
                        result[field] = pd.cut(numeric, bins=10, duplicates="drop")
        return result
