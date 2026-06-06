from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log10, sqrt
from typing import Any

import pandas as pd

from src.services.rendering.axis_domain_policy import AxisDomainPolicy
from src.services.rendering.chart_render_policy_types import ChartRenderPolicyResult
from src.services.rendering.label_fit_policy import LabelFitPolicy
from src.services.rendering.spec_traversal import iter_unit_specs
from src.services.rendering.noninformative_mark_policy import NonInformativeMarkPolicy


@dataclass(frozen=True)
class ChartRenderPolicyOptions:
    target: str = "artifact"
    export_scale: float | None = None
    min_width: int = 280
    max_width: int = 860
    min_height: int = 260
    max_height: int = 620
    default_dpi: int = 192


class ChartRenderPolicy:
    """Compute compact, readable Vega-Lite canvas and PNG scale from spec structure and data.

    The policy is intentionally generic: it estimates visual density from encodings, paneling,
    category cardinality, label length, legends and row count instead of hard-coding individual
    chart types. Logical width/height control layout compactness; scale controls PNG quality.
    """

    DISCRETE_TYPES = {"nominal", "ordinal"}
    TEMPORAL_TYPES = {"temporal"}
    QUANTITATIVE_TYPES = {"quantitative"}
    LEGEND_CHANNELS = {"color", "shape", "fill", "stroke", "opacity", "size"}

    @classmethod
    def compute(
            cls,
            *,
            spec: dict[str, Any],
            data: pd.DataFrame | None = None,
            target: str = "artifact",
            default_dpi: int = 192,
            export_scale: float | None = None,
    ) -> ChartRenderPolicyResult:
        options = ChartRenderPolicyOptions(target=target, default_dpi=default_dpi, export_scale=export_scale)
        structures = list(iter_unit_specs(spec)) or [spec]
        panel_columns, panel_rows = cls._panel_shape(spec)
        panel_count = max(1, panel_columns * panel_rows)
        encodings = [cls._encoding(unit) for unit in structures]
        axis_stats = cls._axis_stats(encodings, data)
        visual_units = cls._visual_unit_count(axis_stats, data)
        label_budget = cls._label_budget(axis_stats)
        legend_budget = cls._legend_budget(encodings, data)
        density_budget = cls._density_budget(axis_stats, data)
        panel_width_factor = 1.0 / max(1.0, sqrt(panel_count) * 0.72)
        panel_height_factor = 1.0 / max(1.0, sqrt(panel_count) * 0.82)

        width = 220 + visual_units * cls._unit_width(axis_stats) + label_budget + legend_budget + density_budget
        height = 230 + cls._vertical_budget(axis_stats, data) + min(120, 24 * max(0, panel_rows - 1))
        if cls._has_vertical_composition(spec):
            height += 120
        if cls._has_horizontal_composition(spec):
            width += 140

        width = int(ceil(width * panel_width_factor / 10.0) * 10)
        height = int(ceil(height * panel_height_factor / 10.0) * 10)
        width = cls._clamp(width, options.min_width, options.max_width)
        height = cls._clamp(height, options.min_height, options.max_height)

        scale = cls._scale_for_target(target=options.target, default_dpi=options.default_dpi, explicit_scale=options.export_scale)
        axis_config = cls._axis_config(axis_stats)
        legend_config = cls._legend_config(legend_budget)
        padding = cls._padding(axis_stats, width=width, height=height)
        reasoning = [
            f"visual_units={visual_units}",
            f"label_budget={label_budget}",
            f"legend_budget={legend_budget}",
            f"panel_shape={panel_columns}x{panel_rows}",
            f"scale={scale}",
        ]
        return ChartRenderPolicyResult(
            width=width,
            height=height,
            scale=scale,
            autosize={"type": "pad", "contains": "padding"},
            axis_config=axis_config,
            legend_config=legend_config,
            padding=padding,
            reasoning=reasoning,
        )

    @classmethod
    def apply(
            cls,
            spec: dict[str, Any],
            *,
            data: pd.DataFrame | None = None,
            target: str = "artifact",
            default_dpi: int = 192,
            export_scale: float | None = None,
    ) -> tuple[dict[str, Any], ChartRenderPolicyResult]:
        clone = cls._deepcopy_jsonish(spec)
        semantic_result = AxisDomainPolicy.apply(clone, data=data)
        noninformative_result = NonInformativeMarkPolicy.apply(clone, data=data)
        result = cls.compute(spec=clone, data=data, target=target, default_dpi=default_dpi, export_scale=export_scale)
        label_result = LabelFitPolicy.apply(clone, data=data, render_policy=result)
        padding = {
            key: max(int(result.padding.get(key, 0)), int(label_result.padding.get(key, 0)))
            for key in {"left", "right", "top", "bottom"}
        }
        reasoning = [*result.reasoning, *semantic_result.changes, *noninformative_result.changes, *label_result.changes]
        result = ChartRenderPolicyResult(
            width=result.width,
            height=result.height,
            scale=result.scale,
            autosize=result.autosize,
            axis_config=result.axis_config,
            legend_config=result.legend_config,
            padding=padding,
            reasoning=reasoning,
        )
        clone["width"] = result.width
        clone["height"] = result.height
        clone["autosize"] = result.autosize
        clone["padding"] = result.padding
        config = clone.setdefault("config", {})
        axis_config = config.setdefault("axis", {})
        for key, value in result.axis_config.items():
            axis_config.setdefault(key, value)
        legend_config = config.setdefault("legend", {})
        for key, value in result.legend_config.items():
            legend_config.setdefault(key, value)
        return clone, result

    @staticmethod
    def _deepcopy_jsonish(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: ChartRenderPolicy._deepcopy_jsonish(item) for key, item in value.items()}
        if isinstance(value, list):
            return [ChartRenderPolicy._deepcopy_jsonish(item) for item in value]
        return value

    @staticmethod
    def _encoding(spec: dict[str, Any]) -> dict[str, Any]:
        value = spec.get("encoding")
        return value if isinstance(value, dict) else {}

    @classmethod
    def _axis_stats(cls, encodings: list[dict[str, Any]], data: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for channel in ("x", "y"):
            channel_defs = [encoding.get(channel) for encoding in encodings if isinstance(encoding.get(channel), dict)]
            field_defs = [item for item in channel_defs if isinstance(item, dict)]
            field = cls._first_field(field_defs)
            type_name = cls._first_type(field_defs)
            cardinality, longest_label = cls._field_cardinality_and_label(field, data)
            stats[channel] = {
                "field": field,
                "type": type_name,
                "cardinality": cardinality,
                "longest_label": longest_label,
                "is_discrete": cls._is_discrete(type_name, field, data),
                "is_temporal": type_name in cls.TEMPORAL_TYPES,
                "is_quantitative": type_name in cls.QUANTITATIVE_TYPES,
            }
        return stats

    @staticmethod
    def _first_field(definitions: list[dict[str, Any]]) -> str | None:
        for definition in definitions:
            field = definition.get("field")
            if isinstance(field, str) and field.strip():
                return field
        return None

    @staticmethod
    def _first_type(definitions: list[dict[str, Any]]) -> str | None:
        for definition in definitions:
            type_name = definition.get("type")
            if isinstance(type_name, str) and type_name.strip():
                return type_name.strip().lower()
        return None

    @classmethod
    def _is_discrete(cls, type_name: str | None, field: str | None, data: pd.DataFrame | None) -> bool:
        if type_name in cls.DISCRETE_TYPES:
            return True
        if not field or data is None or field not in data.columns:
            return False
        dtype = data[field].dtype
        return bool(
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        )

    @staticmethod
    def _field_cardinality_and_label(field: str | None, data: pd.DataFrame | None) -> tuple[int, int]:
        if not field or data is None or field not in data.columns:
            return 1, len(str(field or ""))
        series = data[field].dropna()
        if series.empty:
            return 1, len(str(field))
        values = series.astype(str)
        cardinality = int(values.nunique(dropna=True))
        longest = int(values.map(len).max()) if len(values) else len(str(field))
        return max(1, cardinality), max(len(str(field)), longest)

    @classmethod
    def _visual_unit_count(cls, axis_stats: dict[str, dict[str, Any]], data: pd.DataFrame | None) -> int:
        x = axis_stats.get("x", {})
        y = axis_stats.get("y", {})
        if x.get("is_discrete"):
            return cls._clamp(int(x.get("cardinality", 1)), 1, 28)
        if y.get("is_discrete"):
            return 1
        row_count = 0 if data is None else int(len(data))
        if row_count <= 0:
            return 8
        return cls._clamp(int(ceil(sqrt(row_count))), 5, 32)

    @classmethod
    def _unit_width(cls, axis_stats: dict[str, dict[str, Any]]) -> int:
        x = axis_stats.get("x", {})
        y = axis_stats.get("y", {})
        if x.get("is_discrete"):
            return 54
        if y.get("is_discrete"):
            return 34
        if x.get("is_temporal"):
            return 18
        return 16

    @classmethod
    def _label_budget(cls, axis_stats: dict[str, dict[str, Any]]) -> int:
        x = axis_stats.get("x", {})
        y = axis_stats.get("y", {})
        budget = 0
        if x.get("is_discrete"):
            budget += cls._clamp(int(x.get("longest_label", 0)) * 5, 0, 150)
        if y.get("is_discrete"):
            # Left-side labels need margin, not wider plot area. Keep this small and let padding handle text.
            budget += cls._clamp(int(y.get("longest_label", 0)) * 2, 0, 80)
        return budget

    @classmethod
    def _legend_budget(cls, encodings: list[dict[str, Any]], data: pd.DataFrame | None) -> int:
        budget = 0
        for encoding in encodings:
            for channel in cls.LEGEND_CHANNELS:
                definition = encoding.get(channel)
                if not isinstance(definition, dict):
                    continue
                if not cls._legend_enabled(definition):
                    continue
                field = definition.get("field") if isinstance(definition.get("field"), str) else None
                cardinality, longest = cls._field_cardinality_and_label(field, data)
                budget += cls._clamp(20 + min(cardinality, 8) * 5 + longest * 3, 30, 160)
        return cls._clamp(budget, 0, 190)

    @staticmethod
    def _density_budget(axis_stats: dict[str, dict[str, Any]], data: pd.DataFrame | None) -> int:
        if data is None:
            return 0
        row_count = len(data)
        if row_count <= 50:
            return 0
        if axis_stats.get("x", {}).get("is_discrete") or axis_stats.get("y", {}).get("is_discrete"):
            return min(80, int(log10(row_count) * 24))
        return min(160, int(log10(row_count) * 42))

    @classmethod
    def _vertical_budget(cls, axis_stats: dict[str, dict[str, Any]], data: pd.DataFrame | None) -> int:
        x = axis_stats.get("x", {})
        y = axis_stats.get("y", {})
        row_count = 0 if data is None else len(data)
        budget = 0
        if y.get("is_discrete"):
            budget += cls._clamp(int(y.get("cardinality", 1)) * 28, 20, 260)
        else:
            budget += 90
        if x.get("is_discrete"):
            budget += cls._clamp(int(x.get("longest_label", 0)) * 3, 0, 100)
        if row_count > 200 and not y.get("is_discrete"):
            budget += min(80, int(log10(row_count) * 22))
        return budget

    @classmethod
    def _panel_shape(cls, spec: dict[str, Any]) -> tuple[int, int]:
        columns = 1
        rows = 1
        repeat = spec.get("repeat")
        if isinstance(repeat, dict):
            columns = max(columns, cls._list_len(repeat.get("column")))
            rows = max(rows, cls._list_len(repeat.get("row")))
        elif isinstance(repeat, list):
            columns = max(columns, len(repeat))
        facet = spec.get("facet")
        if isinstance(facet, dict):
            columns = max(columns, cls._facet_cardinality(facet.get("column")))
            rows = max(rows, cls._facet_cardinality(facet.get("row")))
        columns = max(columns, cls._list_len(spec.get("hconcat")))
        rows = max(rows, cls._list_len(spec.get("vconcat")))
        return max(1, columns), max(1, rows)

    @staticmethod
    def _list_len(value: Any) -> int:
        return len(value) if isinstance(value, list) and value else 1

    @staticmethod
    def _facet_cardinality(value: Any) -> int:
        if isinstance(value, dict):
            return 2
        return 1

    @staticmethod
    def _has_vertical_composition(spec: dict[str, Any]) -> bool:
        return isinstance(spec.get("vconcat"), list) or (isinstance(spec.get("facet"), dict) and "row" in spec.get("facet", {}))

    @staticmethod
    def _has_horizontal_composition(spec: dict[str, Any]) -> bool:
        return isinstance(spec.get("hconcat"), list) or (isinstance(spec.get("facet"), dict) and "column" in spec.get("facet", {}))

    @staticmethod
    def _axis_config(axis_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
        longest = max(int(stats.get("longest_label", 0)) for stats in axis_stats.values())
        return {
            "labelLimit": max(180, min(420, 12 * longest)),
            "labelOverlap": "greedy",
            "labelBound": True,
            "titleLimit": 300,
            "labelFontSize": 11,
            "titleFontSize": 12,
        }

    @classmethod
    def _padding(cls, axis_stats: dict[str, dict[str, Any]], *, width: int, height: int) -> dict[str, int]:
        x = axis_stats.get("x", {})
        y = axis_stats.get("y", {})
        bottom = 40
        left = 54
        if x.get("is_discrete"):
            x_longest = int(x.get("longest_label", 0))
            x_categories = max(1, int(x.get("cardinality", 1)))
            slot_width = max(1.0, width / x_categories)
            estimated_label_width = x_longest * 6.4
            if estimated_label_width > slot_width * 1.45:
                bottom = cls._clamp(40 + x_longest * 5, 80, 190)
            elif estimated_label_width > slot_width * 0.95:
                bottom = cls._clamp(34 + x_longest * 3, 56, 120)
            else:
                bottom = cls._clamp(32 + x_longest, 38, 68)
        if y.get("is_discrete"):
            y_longest = int(y.get("longest_label", 0))
            left = cls._clamp(46 + y_longest * 6, 64, 230)
        return {"left": left, "right": 24, "top": 36, "bottom": bottom}

    @staticmethod
    def _legend_config(legend_budget: int) -> dict[str, Any]:
        return {"labelLimit": max(180, min(360, legend_budget + 120)), "titleLimit": 300}

    @staticmethod
    def _legend_enabled(definition: dict[str, Any]) -> bool:
        if "legend" not in definition:
            return True
        legend = definition.get("legend")
        return legend is not None and legend is not False

    @staticmethod
    def _scale_for_target(*, target: str, default_dpi: int, explicit_scale: float | None) -> float:
        if explicit_scale is not None:
            return max(1.0, float(explicit_scale))
        normalized = (target or "artifact").strip().lower()
        if normalized == "publication":
            return 3.0
        if normalized in {"artifact", "benchmark", "vlm"}:
            return max(2.0, round(max(default_dpi, 192) / 96.0, 2))
        return max(1.0, round(max(default_dpi, 96) / 96.0, 2))

    @staticmethod
    def _clamp(value: int, low: int, high: int) -> int:
        return max(low, min(int(value), high))
