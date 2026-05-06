from __future__ import annotations

import struct
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.domain.models import PlotImageArtifact, PlotRenderingResult, SpecValidationResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.data import read_dataframe


class VegaLitePlotDrawingService(BaseService):
    """Render validated Vega-Lite specs with the Vega runtime, not a Matplotlib subset."""

    def invoke(self, spec_validation: SpecValidationResult, run_id: str,
               runtime: RuntimeContext) -> PlotRenderingResult:
        if not spec_validation.is_valid:
            raise RuntimeError('Cannot draw a Vega-Lite plot from an invalid specification.')
        try:
            import vl_convert as vlc  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "vl-convert-python is required for Vega-Lite rendering. Install dependency 'vl-convert-python'."
            ) from exc

        spec = deepcopy(spec_validation.validated_spec)
        data_url = self._extract_data_url(spec)
        df = read_dataframe(data_url)
        spec_with_data = self._spec_add_data(spec, df)
        spec_with_data = self._apply_render_defaults(spec_with_data)

        run_dir = runtime.ensure_run_dir(run_id)
        image_path = runtime.next_artifact_path('plot.png', run_id=run_id)
        runtime.save_json_artifact('artifacts/vega_lite_validated_spec.json', spec, run_id=run_id, numbered=True)
        runtime.save_json_artifact('artifacts/vega_lite_render_spec.json', spec_with_data, run_id=run_id, numbered=True)

        try:
            png_bytes = vlc.vegalite_to_png(vl_spec=spec_with_data, scale=1)
            image_path.write_bytes(png_bytes)
            scenegraph = vlc.vegalite_to_scenegraph(vl_spec=spec_with_data, show_warnings=False)
        except Exception as exc:
            raise RuntimeError(f'Vega-Lite rendering failed: {exc}') from exc

        pixel_width, pixel_height = self._read_png_size(image_path)
        scenegraph_summary = self._summarize_scenegraph(scenegraph)
        scenegraph_summary['source'] = 'vl-convert-python'
        scenegraph_summary['notes'].append('Rendered with Vega-Lite runtime via vl-convert-python.')
        runtime.save_json_artifact('artifacts/rendered_scenegraph.json', scenegraph_summary, run_id=run_id, numbered=True)
        runtime.save_json_artifact('artifacts/rendered_scenegraph_raw.json', scenegraph, run_id=run_id, numbered=True)

        return PlotRenderingResult(
            plot_image=PlotImageArtifact(image_path=image_path.as_posix(), width=pixel_width, height=pixel_height),
            rendered_scenegraph=scenegraph_summary,
            render_notes=scenegraph_summary['notes'],
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
    def _apply_render_defaults(spec: dict[str, Any]) -> dict[str, Any]:
        clone = deepcopy(spec)
        clone.setdefault('width', 720)
        clone.setdefault('height', 420)
        config = clone.setdefault('config', {})
        axis_config = config.setdefault('axis', {})
        axis_config.setdefault('labelLimit', 180)
        axis_config.setdefault('labelOverlap', 'greedy')
        axis_config.setdefault('titleLimit', 220)
        axis_config.setdefault('labelFontSize', 11)
        axis_config.setdefault('titleFontSize', 12)
        legend_config = config.setdefault('legend', {})
        legend_config.setdefault('labelLimit', 180)
        return clone

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
        marks_count = sum(cls._count_mark_items(node) for node in mark_nodes)
        mark_types = sorted({str(node.get('marktype')) for node in mark_nodes if node.get('marktype')})
        return {
            'mark_type': ','.join(mark_types) if mark_types else 'unknown',
            'mark_types': mark_types,
            'marks_count': int(marks_count),
            'axes': [str(node.get('ariaRoleDescription') or node.get('orient') or 'axis') for node in axis_nodes],
            'has_legend': bool(legend_nodes),
            'legend_count': len(legend_nodes),
            'notes': [],
        }

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
