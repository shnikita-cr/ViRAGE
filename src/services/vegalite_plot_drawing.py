from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.domain.models import PlotImageArtifact, PlotRenderingResult, SpecValidationResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class VegaLitePlotDrawingService(BaseService):
    def invoke(self, spec_validation: SpecValidationResult, run_id: str,
               runtime: RuntimeContext) -> PlotRenderingResult:
        if not spec_validation.is_valid:
            raise RuntimeError('Cannot draw a Vega-Lite plot from an invalid specification.')
        spec = spec_validation.validated_spec
        data_url = spec['data']['url']
        df = pd.read_csv(data_url)
        df = self._ensure_unique_columns(df)
        df = self._apply_transforms(df, spec.get('transform', []))
        run_dir = runtime.ensure_run_dir(run_id)
        image_path = run_dir / 'plot.png'
        runtime.save_json_artifact('artifacts/vega_lite_validated_spec.json', spec, run_id=run_id)

        width = int(spec.get('width', 8 * runtime.settings.default_figure_dpi))
        height = int(spec.get('height', 5 * runtime.settings.default_figure_dpi))
        fig, ax = plt.subplots(
            figsize=(max(width, 320) / runtime.settings.default_figure_dpi,
                     max(height, 240) / runtime.settings.default_figure_dpi),
            dpi=runtime.settings.default_figure_dpi,
        )
        mark = spec.get('mark')
        mark_type = mark.get('type') if isinstance(mark, dict) else mark
        encoding = spec.get('encoding', {})
        x = encoding.get('x', {}) if isinstance(encoding.get('x'), dict) else {}
        y = encoding.get('y', {}) if isinstance(encoding.get('y'), dict) else {}
        color = encoding.get('color', {}) if isinstance(encoding.get('color'), dict) else {}
        x_field = x.get('field')
        y_field = y.get('field')
        color_field = color.get('field')
        aggregate = y.get('aggregate')
        marks_count = 0
        notes: list[str] = []

        if x.get('type') == 'temporal' and x_field in df.columns:
            df[x_field] = pd.to_datetime(df[x_field], errors='coerce')
        if y_field in df.columns:
            df[y_field] = pd.to_numeric(df[y_field], errors='coerce')
        if color_field in df.columns and color.get('type') == 'quantitative':
            df[color_field] = pd.to_numeric(df[color_field], errors='coerce')

        if mark_type in {'line', 'area'}:
            plot_df, y_plot_field = self._aggregate_for_plot(df, x_field=x_field, y_field=y_field,
                                                             color_field=color_field, aggregate=aggregate,
                                                             dropna=[x_field,
                                                                     y_field if aggregate != 'count' else None])
            plot_df = plot_df.sort_values(self._unique_preserve(
                [color_field if color_field in plot_df.columns else None,
                 x_field if x_field in plot_df.columns else None]))
            if color_field and color_field in plot_df.columns and color_field != x_field:
                for label, frame in plot_df.groupby(color_field, dropna=False):
                    ax.plot(frame[x_field], frame[y_plot_field], marker='o', label=str(label))
                    if mark_type == 'area':
                        ax.fill_between(frame[x_field], frame[y_plot_field], alpha=0.2)
                    marks_count += int(len(frame))
                ax.legend()
            else:
                ax.plot(plot_df[x_field], plot_df[y_plot_field], marker='o')
                if mark_type == 'area':
                    ax.fill_between(plot_df[x_field], plot_df[y_plot_field], alpha=0.2)
                marks_count = int(len(plot_df))
        elif mark_type == 'bar':
            plot_df, y_plot_field = self._aggregate_for_plot(df, x_field=x_field, y_field=y_field,
                                                             color_field=color_field, aggregate=aggregate,
                                                             dropna=[x_field,
                                                                     y_field if aggregate != 'count' else None])
            if color_field and color_field in plot_df.columns and color_field != x_field:
                pivot = plot_df.pivot(index=x_field, columns=color_field, values=y_plot_field).fillna(0)
                pivot.plot(kind='bar', ax=ax)
                marks_count = int(pivot.size)
            else:
                ax.bar(plot_df[x_field].astype(str), plot_df[y_plot_field])
                marks_count = int(len(plot_df))
        elif mark_type in {'point', 'circle', 'tick'}:
            plot_df, y_plot_field = self._aggregate_for_plot(df, x_field=x_field, y_field=y_field,
                                                             color_field=color_field, aggregate=aggregate,
                                                             dropna=[x_field,
                                                                     y_field if aggregate != 'count' else None])
            if color_field and color_field in plot_df.columns and color_field != x_field:
                for label, frame in plot_df.groupby(color_field, dropna=False):
                    ax.scatter(frame[x_field], frame[y_plot_field], label=str(label))
                    marks_count += int(len(frame))
                ax.legend()
            else:
                ax.scatter(plot_df[x_field], plot_df[y_plot_field])
                marks_count = int(len(plot_df))
        elif mark_type == 'histogram':
            series = pd.to_numeric(df[x_field], errors='coerce').dropna()
            ax.hist(series, bins=10)
            marks_count = int(len(series))
        elif mark_type == 'boxplot':
            if color_field and color_field in df.columns and color_field != y_field:
                groups = []
                labels = []
                for label, frame in df.groupby(color_field, dropna=False):
                    series = pd.to_numeric(frame[y_field], errors='coerce').dropna()
                    if not series.empty:
                        groups.append(series)
                        labels.append(str(label))
                if groups:
                    ax.boxplot(groups, tick_labels=labels)
                    marks_count = sum(len(group) for group in groups)
            else:
                series = pd.to_numeric(df[y_field], errors='coerce').dropna()
                ax.boxplot(series)
                marks_count = int(len(series))
        else:
            raise RuntimeError(f'Unsupported rendered mark type: {mark_type}')

        ax.set_title(spec.get('title') or 'Generated visualization')
        if x_field:
            ax.set_xlabel(x_field)
        if y_field or aggregate == 'count':
            ax.set_ylabel(y.get('title') if isinstance(y, dict) and y.get('title') else (y_field or 'Count'))
        if x.get('type') == 'temporal':
            fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(image_path)
        pixel_width, pixel_height = map(int, fig.get_size_inches() * fig.dpi)
        plt.close(fig)

        scenegraph = {
            'mark_type': mark_type,
            'marks_count': marks_count,
            'axes': [axis for axis in [x_field, y_field or ('__count__' if aggregate == 'count' else None)] if axis],
            'has_legend': bool(color_field and color_field != x_field),
            'notes': notes,
        }
        runtime.save_json_artifact('artifacts/rendered_scenegraph.json', scenegraph, run_id=run_id)
        return PlotRenderingResult(
            plot_image=PlotImageArtifact(image_path=image_path.as_posix(), width=pixel_width, height=pixel_height),
            rendered_scenegraph=scenegraph,
            render_notes=notes,
        )

    @staticmethod
    def _unique_preserve(values: list[str | None]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if not value:
                continue
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    @classmethod
    def _prepare_frame(cls, df: pd.DataFrame, columns: list[str | None], *, dropna: list[str | None]) -> pd.DataFrame:
        selected = cls._unique_preserve(columns)
        required = cls._unique_preserve(dropna)
        return df[selected].dropna(subset=required).copy()

    @classmethod
    def _aggregate_for_plot(
            cls,
            df: pd.DataFrame,
            *,
            x_field: str | None,
            y_field: str | None,
            color_field: str | None,
            aggregate: str | None,
            dropna: list[str | None],
    ) -> tuple[pd.DataFrame, str]:
        if aggregate == 'count':
            grouping = cls._unique_preserve([x_field, color_field if color_field != x_field else None])
            if not grouping:
                return pd.DataFrame({'__count__': [len(df)]}), '__count__'
            base = cls._prepare_frame(df, grouping, dropna=[x_field])
            grouped = base.groupby(grouping, dropna=False).size().reset_index(name='__count__')
            return grouped, '__count__'

        columns = cls._unique_preserve([x_field, y_field, color_field if color_field != x_field else None])
        base = cls._prepare_frame(df, columns, dropna=dropna)
        if aggregate and x_field and y_field:
            grouping = cls._unique_preserve([x_field, color_field if color_field != x_field else None])
            grouped = base.groupby(grouping, dropna=False, as_index=False)[y_field].agg(aggregate)
            return grouped, y_field
        return base, y_field or '__value__'

    @staticmethod
    def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
        if not df.columns.duplicated().any():
            return df
        counts: dict[str, int] = {}
        new_columns: list[str] = []
        for name in df.columns:
            counts[name] = counts.get(name, 0) + 1
            if counts[name] == 1:
                new_columns.append(name)
            else:
                new_columns.append(f'{name}__dup{counts[name] - 1}')
        copy = df.copy()
        copy.columns = new_columns
        return copy

    @staticmethod
    def _apply_transforms(df: pd.DataFrame, transforms: list[dict[str, Any]]) -> pd.DataFrame:
        result = df.copy()
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            kind = transform.get('kind')
            if kind == 'filter':
                expression = transform.get('filter') or transform.get('expression')
                if isinstance(expression, str) and expression.strip():
                    try:
                        result = result.query(expression)
                    except Exception:
                        continue
            elif kind == 'calculate':
                as_name = transform.get('as') or transform.get('field_name')
                expression = transform.get('calculate') or transform.get('expression')
                if isinstance(as_name, str) and isinstance(expression, str) and expression.strip():
                    try:
                        result[as_name] = result.eval(expression)
                    except Exception:
                        continue
            elif kind == 'bin':
                field = transform.get('field') or transform.get('field_name')
                if isinstance(field, str) and field in result.columns:
                    numeric = pd.to_numeric(result[field], errors='coerce')
                    if numeric.notna().sum() >= 2:
                        result[field] = pd.cut(numeric, bins=10, duplicates='drop')
        return result
