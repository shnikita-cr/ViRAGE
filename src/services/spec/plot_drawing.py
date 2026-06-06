from __future__ import annotations

import struct
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.domain.models import PlotImageArtifact, PlotRenderingResult, SpecValidationResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.data import read_dataframe
from src.services.chart_quality import ChartQualityEvaluator, ChartQualityPipeline, ChartQualityThresholds
from src.services.rendering import ChartRenderPolicy
from src.services.spec.data_injection import spec_with_inline_data


@dataclass(frozen=True)
class _Bounds:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def clipped_by(self, *, width: float, height: float) -> bool:
        return self.x1 < 0 or self.y1 < 0 or self.x2 > width or self.y2 > height


class VegaLitePlotDrawingService(BaseService):
    """Render validated Vega-Lite specs with the Vega runtime."""

    def __init__(self) -> None:
        self.quality_pipeline = ChartQualityPipeline()

    def invoke(self, spec_validation: SpecValidationResult, run_id: str,
               runtime: RuntimeContext) -> PlotRenderingResult:
        if not spec_validation.is_valid:
            raise RuntimeError('Cannot draw a Vega-Lite plot from an invalid specification.')
        try:
            import vl_convert as vlc  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "vl-convert-python is required for Vega-Lite rendering. Install dependency 'vl-convert-python'."
            ) from exc

        spec = deepcopy(spec_validation.validated_spec)
        data_url = self._extract_data_url(spec)
        df = read_dataframe(data_url)
        spec_with_data = spec_with_inline_data(spec, df)
        quality_pipeline = self.quality_pipeline.apply(spec_with_data, data=df)
        spec_with_data, render_policy = self._apply_render_policy(
            quality_pipeline.spec,
            df=df,
            default_dpi=int(getattr(runtime.settings, "default_figure_dpi", 192) or 192),
            export_scale=float(getattr(runtime.settings, "vega_export_scale", 2.0) or 2.0),
        )

        run_dir = runtime.ensure_run_dir(run_id)
        image_path = runtime.next_artifact_path('plot.png', run_id=run_id)

        try:
            png_bytes = vlc.vegalite_to_png(vl_spec=spec_with_data, scale=render_policy.scale)
            image_path.write_bytes(png_bytes)
            scenegraph = vlc.vegalite_to_scenegraph(vl_spec=spec_with_data, show_warnings=False)
        except (ValueError, RuntimeError, OSError) as exc:
            raise RuntimeError(f'Vega-Lite rendering failed: {exc}') from exc

        pixel_width, pixel_height = self._read_png_size(image_path)
        scenegraph_summary = self._summarize_scenegraph(scenegraph)
        scenegraph_summary['source'] = 'vl-convert-python'
        scenegraph_summary['render_policy'] = {
            'width': render_policy.width,
            'height': render_policy.height,
            'scale': render_policy.scale,
            'padding': render_policy.padding,
            'reasoning': render_policy.reasoning,
        }
        scenegraph_summary['quality_policy'] = {
            'changes': quality_pipeline.changes,
            'issues': quality_pipeline.issue_dicts(),
        }
        quality_report = ChartQualityEvaluator(ChartQualityThresholds.from_settings(runtime.settings)).evaluate(
            spec=spec_with_data,
            png_path=image_path,
            scenegraph_summary=scenegraph_summary,
            data=df,
            policy_issues=quality_pipeline.issues,
        )
        scenegraph_summary['chart_quality'] = quality_report.to_dict()
        scenegraph_summary['notes'].append('Rendered with Vega-Lite runtime via vl-convert-python.')
        if quality_pipeline.changes:
            scenegraph_summary['notes'].append('Applied chart quality policies: ' + ', '.join(quality_pipeline.changes[:8]))
        if quality_report.status != 'pass':
            scenegraph_summary['notes'].append(
                'Chart quality status: ' + quality_report.status + '; issues=' + ', '.join(
                    issue.code for issue in quality_report.issues[:8]
                )
            )

        return PlotRenderingResult(
            plot_image=PlotImageArtifact(image_path=image_path.as_posix(), width=pixel_width, height=pixel_height),
            rendered_scenegraph=scenegraph_summary,
            render_notes=scenegraph_summary['notes'],
            chart_quality_report=quality_report.to_dict(),
        )

    @staticmethod
    def _extract_data_url(spec: dict[str, Any]) -> str:
        data = spec.get('data')
        if not isinstance(data, dict):
            raise RuntimeError('Validated Vega-Lite spec must contain data.url.')
        data_url = data.get('url')
        if not isinstance(data_url, str) or not data_url.strip():
            raise RuntimeError('Validated Vega-Lite spec must contain a non-empty data.url.')
        return data_url

    @staticmethod
    def _spec_add_data(spec: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        clone = deepcopy(spec)
        clone['data'] = {'values': df.where(pd.notna(df), None).to_dict(orient='records')}
        return clone

    @staticmethod
    def _apply_render_policy(
            spec: dict[str, Any],
            *,
            df: pd.DataFrame,
            default_dpi: int,
            export_scale: float,
    ):
        return ChartRenderPolicy.apply(
            spec,
            data=df,
            target='artifact',
            default_dpi=default_dpi,
            export_scale=export_scale,
        )

    @staticmethod
    def _walk(node: Any):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from VegaLitePlotDrawingService._walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from VegaLitePlotDrawingService._walk(item)

    @classmethod
    def _summarize_scenegraph(cls, scenegraph: dict[str, Any]) -> dict[str, Any]:
        mark_nodes = [node for node in cls._walk(scenegraph) if node.get('role') == 'mark']
        axis_nodes = [node for node in cls._walk(scenegraph) if node.get('role') == 'axis']
        legend_nodes = [node for node in cls._walk(scenegraph) if node.get('role') == 'legend']
        text_nodes = [node for node in cls._walk(scenegraph) if cls._node_is_text(node)]
        width = float(scenegraph.get('width') or 0.0)
        height = float(scenegraph.get('height') or 0.0)
        mark_bounds = cls._merged_bounds(cls._node_bounds(node) for node in mark_nodes)
        text_bounds = [bounds for bounds in (cls._node_bounds(node) for node in text_nodes) if bounds is not None]
        marks_count = sum(cls._count_mark_items(node) for node in mark_nodes)
        mark_types = sorted({str(node.get('marktype')) for node in mark_nodes if node.get('marktype')})
        return {
            'mark_type': ','.join(mark_types) if mark_types else 'unknown',
            'mark_types': mark_types,
            'marks_count': int(marks_count),
            'axes': [str(node.get('ariaRoleDescription') or node.get('orient') or 'axis') for node in axis_nodes],
            'has_legend': bool(legend_nodes),
            'legend_count': len(legend_nodes),
            'scenegraph_width': width,
            'scenegraph_height': height,
            'plot_area_usage': cls._plot_area_usage(mark_bounds, width=width, height=height),
            'mark_bbox_area': 0.0 if mark_bounds is None else mark_bounds.area,
            'text_count': len(text_bounds),
            'clipped_text_count': cls._clipped_text_count(text_bounds, width=width, height=height),
            'notes': [],
        }


    @staticmethod
    def _node_is_text(node: dict[str, Any]) -> bool:
        return node.get('marktype') == 'text' or node.get('role') in {'axis-label', 'axis-title', 'legend-label', 'legend-title', 'title'}

    @classmethod
    def _node_bounds(cls, node: dict[str, Any]) -> _Bounds | None:
        bounds = cls._bounds_from_mapping(node.get('bounds'))
        if bounds is not None:
            return bounds
        return cls._merged_bounds(cls._bounds_from_mapping(item.get('bounds')) for item in node.get('items', []) if isinstance(item, dict))

    @staticmethod
    def _bounds_from_mapping(value: Any) -> _Bounds | None:
        if not isinstance(value, dict):
            return None
        keys = {'x1', 'y1', 'x2', 'y2'}
        if not keys.issubset(value):
            return None
        return _Bounds(float(value['x1']), float(value['y1']), float(value['x2']), float(value['y2']))

    @staticmethod
    def _merged_bounds(bounds_values) -> _Bounds | None:
        bounds = [value for value in bounds_values if value is not None]
        if not bounds:
            return None
        return _Bounds(
            min(value.x1 for value in bounds),
            min(value.y1 for value in bounds),
            max(value.x2 for value in bounds),
            max(value.y2 for value in bounds),
        )

    @staticmethod
    def _plot_area_usage(bounds: _Bounds | None, *, width: float, height: float) -> float:
        if bounds is None or width <= 0 or height <= 0:
            return 0.0
        return max(0.0, min(1.0, bounds.area / (width * height)))

    @staticmethod
    def _clipped_text_count(bounds: list[_Bounds], *, width: float, height: float) -> int:
        if width <= 0 or height <= 0:
            return 0
        return sum(1 for item in bounds if item.clipped_by(width=width, height=height))

    @classmethod
    def _count_mark_items(cls, mark_node: dict[str, Any]) -> int:
        items = mark_node.get('items')
        if not isinstance(items, list):
            return 0
        if mark_node.get('marktype') in {'line', 'area', 'trail'}:
            return 0 if len(items) <= 1 and all(not item.get('defined', True) for item in items) else len(items)
        return sum(1 for item in items if cls._is_visible_mark_item(item))

    @staticmethod
    def _is_visible_mark_item(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        if item.get('opacity', 1) == 0:
            return False
        if item.get('defined') is False:
            return False
        if item.get('width', 1) == 0 or item.get('height', 1) == 0 or item.get('size', 1) == 0:
            return False
        return True

    @staticmethod
    def _read_png_size(path: Path) -> tuple[int, int]:
        with path.open('rb') as file:
            header = file.read(24)
        if len(header) >= 24 and header[:8] == b'\x89PNG\r\n\x1a\n':
            width, height = struct.unpack('>II', header[16:24])
            return int(width), int(height)
        return 0, 0
