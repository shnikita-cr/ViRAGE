from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PresentationLabelKey:
    field: str
    aggregate: str = ""
    bin: str = ""
    time_unit: str = ""


@dataclass
class PresentationLabelUse:
    label: str
    priority: int
    path: tuple[Any, ...]
    container_kind: str


@dataclass
class PresentationConsistencyResult:
    spec: dict[str, Any]
    changes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RepeatFieldReference:
    name: str


class SpecPresentationConsistencyService:
    """Deterministic presentation-layer normalization for Vega-Lite specs.

    The service keeps the generated chart structure intact and only normalizes visible text:
    chart title, axis titles, legend titles and tooltip titles. For ordinary fields it builds
    concrete labels from the encoded field and aggregate. For Vega-Lite repeat references it
    keeps the specification parametric: a repeated field is labelled as a generic value while
    row/column headers provide the concrete field names in each repeated view.
    """

    CHANNEL_PRIORITY = {
        "x": 10,
        "y": 10,
        "xOffset": 20,
        "yOffset": 20,
        "color": 30,
        "size": 35,
        "shape": 35,
        "theta": 35,
        "radius": 35,
        "tooltip": 60,
    }

    AGGREGATE_LABELS = {
        "mean": "Average",
        "average": "Average",
        "sum": "Total",
        "count": "Count",
        "valid": "Count",
        "missing": "Missing Count",
        "distinct": "Distinct Count",
        "median": "Median",
        "min": "Minimum",
        "max": "Maximum",
        "q1": "First Quartile",
        "q3": "Third Quartile",
        "ci0": "Confidence Interval Lower Bound",
        "ci1": "Confidence Interval Upper Bound",
        "stderr": "Standard Error",
        "stdev": "Standard Deviation",
        "variance": "Variance",
    }

    TIME_UNIT_LABELS = {
        "year": "Year",
        "quarter": "Quarter",
        "month": "Month",
        "date": "Date",
        "day": "Day",
        "hours": "Hour",
        "minutes": "Minute",
        "seconds": "Second",
        "milliseconds": "Millisecond",
        "yearquarter": "Year-Quarter",
        "yearmonth": "Year-Month",
        "yearmonthdate": "Date",
        "monthdate": "Month-Date",
        "hoursminutes": "Hour-Minute",
        "hoursminutesseconds": "Hour-Minute-Second",
    }

    GENERIC_TITLES = {
        "average", "mean", "avg", "total", "sum", "count", "value", "values", "metric", "metrics",
        "measure", "measurement", "comparison", "trend", "distribution", "chart", "plot", "visualization",
        "relationship", "relationships", "data", "result", "results",
    }
    AGGREGATE_WORDS = {
        "average": "mean",
        "mean": "mean",
        "avg": "mean",
        "total": "sum",
        "sum": "sum",
        "count": "count",
        "median": "median",
        "minimum": "min",
        "min": "min",
        "maximum": "max",
        "max": "max",
    }
    POSITION_CHANNELS = {"x", "y", "x2", "y2"}
    GROUP_CHANNELS = {"color", "shape", "detail", "strokeDash", "row", "column"}
    TOOLTIP_CHANNEL = "tooltip"
    COMPOSITION_KEYS = ("layer", "spec", "concat", "hconcat", "vconcat")

    def normalize(self, spec: dict[str, Any]) -> PresentationConsistencyResult:
        if not isinstance(spec, dict):
            return PresentationConsistencyResult(spec={})

        normalized = deepcopy(spec)
        changes: list[str] = []
        self._normalize_node(normalized, root=normalized, changes=changes)

        # Keep the older consistency pass for cases with nested user-provided titles that do not
        # match the deterministic labels exactly. This is now a fallback, not the main label source.
        label_uses: dict[PresentationLabelKey, list[PresentationLabelUse]] = {}
        self._collect_label_uses(normalized, label_uses, path=())
        replacements: dict[str, str] = {}
        for key, uses in label_uses.items():
            labels = [use.label for use in uses if use.label.strip()]
            unique_labels = self._unique(labels)
            if len(unique_labels) <= 1:
                continue
            canonical = self._choose_canonical_label(uses)
            if not canonical:
                continue
            for label in unique_labels:
                if label != canonical:
                    replacements[label] = canonical
            for use in uses:
                if use.label != canonical:
                    self._set_label(normalized, use.path, canonical, use.container_kind)
            changes.append(
                "Unified presentation labels for "
                f"field={key.field!r}, aggregate={key.aggregate or 'none'}: {unique_labels!r} -> {canonical!r}."
            )

        if replacements:
            self._normalize_titles(normalized, replacements)
        return PresentationConsistencyResult(spec=normalized, changes=self._dedupe(changes))

    @classmethod
    def _normalize_node(cls, node: Any, *, root: dict[str, Any], changes: list[str]) -> None:
        if isinstance(node, dict):
            encoding = node.get("encoding")
            if isinstance(encoding, dict):
                cls._normalize_encoding(encoding, root=root, view=node, changes=changes)
                cls._normalize_view_title(node, encoding, root=root, changes=changes)
            for key in cls.COMPOSITION_KEYS:
                child = node.get(key)
                if isinstance(child, list):
                    for item in child:
                        cls._normalize_node(item, root=root, changes=changes)
                elif isinstance(child, dict):
                    cls._normalize_node(child, root=root, changes=changes)
        elif isinstance(node, list):
            for item in node:
                cls._normalize_node(item, root=root, changes=changes)

    @classmethod
    def _normalize_encoding(
            cls,
            encoding: dict[str, Any],
            *,
            root: dict[str, Any],
            view: dict[str, Any],
            changes: list[str],
    ) -> None:
        mark_type = cls._mark_type(view)
        for channel, channel_def in list(encoding.items()):
            cls._normalize_channel(channel, channel_def, root=root, mark_type=mark_type, changes=changes)
        cls._remove_redundant_color_legend(encoding, changes=changes)

    @classmethod
    def _normalize_channel(
            cls,
            channel: str,
            channel_def: Any,
            *,
            root: dict[str, Any],
            mark_type: str,
            changes: list[str],
    ) -> None:
        if isinstance(channel_def, list):
            for item in channel_def:
                cls._normalize_channel(channel, item, root=root, mark_type=mark_type, changes=changes)
            return
        if not isinstance(channel_def, dict):
            return

        label = cls._label_for_channel(channel_def, channel=channel, root=root, mark_type=mark_type)
        if not label:
            return

        if channel == cls.TOOLTIP_CHANNEL:
            cls._set_channel_title(channel_def, label, changes=changes, channel=channel)
            return

        if channel in cls.POSITION_CHANNELS:
            axis = channel_def.get("axis")
            if axis is None:
                axis = {}
                channel_def["axis"] = axis
            if isinstance(axis, dict):
                cls._set_nested_title(axis, label, changes=changes, owner=f"encoding.{channel}.axis")
                cls._remove_channel_title(channel_def, changes=changes, owner=f"encoding.{channel}")
            return

        if channel in cls.GROUP_CHANNELS:
            legend = channel_def.get("legend")
            if legend is None and channel not in {"detail", "row", "column"}:
                legend = {}
                channel_def["legend"] = legend
            if isinstance(legend, dict):
                cls._set_nested_title(legend, label, changes=changes, owner=f"encoding.{channel}.legend")
                cls._remove_channel_title(channel_def, changes=changes, owner=f"encoding.{channel}")
            elif channel in {"row", "column"}:
                cls._set_channel_title(channel_def, label, changes=changes, channel=channel)
            return

        cls._set_channel_title(channel_def, label, changes=changes, channel=channel)

    @classmethod
    def _normalize_view_title(
            cls,
            view: dict[str, Any],
            encoding: dict[str, Any],
            *,
            root: dict[str, Any],
            changes: list[str],
    ) -> None:
        title = cls._chart_title(encoding, root=root, view=view)
        if not title:
            return
        target_view = root if cls._is_repeat_root(root) and view is not root else view
        current_title = cls._title_text(target_view.get("title"))
        if current_title and not cls._should_replace_chart_title(current_title, encoding, root=root, view=view):
            return
        cls._set_view_title(target_view, title, changes=changes)

    @classmethod
    def _should_replace_chart_title(
            cls,
            title: str,
            encoding: dict[str, Any],
            *,
            root: dict[str, Any],
            view: dict[str, Any],
    ) -> bool:
        normalized = cls._normalize_text(title)
        if not normalized or normalized in cls.GENERIC_TITLES:
            return True
        if cls._title_has_repeated_by_segments(title):
            return True
        mark_type = cls._mark_type(view)
        if mark_type == "boxplot" and "distribution" not in normalized:
            return True
        measure = cls._primary_measure(encoding)
        dimension = cls._primary_dimension(encoding)
        aggregate = cls._aggregate_from_channel(measure) if measure else ""
        measure_field = cls._field_from_channel(measure) if measure else None
        dimension_field = cls._field_from_channel(dimension) if dimension else None
        if aggregate and isinstance(measure_field, str) and cls._normalize_text(measure_field) not in normalized:
            return True
        if aggregate and cls._title_has_wrong_aggregate(normalized, aggregate):
            return True
        if (
                aggregate
                and isinstance(dimension_field, str)
                and cls._normalize_text(dimension_field) in normalized
                and isinstance(measure_field, str)
                and cls._normalize_text(measure_field) not in normalized
        ):
            return True
        deterministic = cls._chart_title(encoding, root=root, view=view)
        if deterministic and cls._normalize_text(deterministic) != normalized:
            if any(word in normalized for word in (" by ", " over ", " vs ")):
                return True
        return False

    @classmethod
    def _title_has_repeated_by_segments(cls, title: str) -> bool:
        parts = [cls._normalize_text(part) for part in re.split(r"\s+by\s+", title, flags=re.IGNORECASE)[1:]]
        return len(parts) != len(set(parts))

    @classmethod
    def _title_has_wrong_aggregate(cls, normalized_title: str, expected_aggregate: str) -> bool:
        expected = cls._canonical_aggregate(expected_aggregate)
        for word, aggregate in cls.AGGREGATE_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", normalized_title) and cls._canonical_aggregate(
                    aggregate) != expected:
                return True
        return False

    @classmethod
    def _chart_title(cls, encoding: dict[str, Any], *, root: dict[str, Any], view: dict[str, Any]) -> str:
        mark_type = cls._mark_type(view)
        x_def = cls._first_channel_def(encoding.get("x"))
        y_def = cls._first_channel_def(encoding.get("y"))
        x_field = cls._field_from_channel(x_def)
        y_field = cls._field_from_channel(y_def)
        x_time = cls._time_unit_from_channel(x_def)
        y_time = cls._time_unit_from_channel(y_def)
        x_is_repeat = isinstance(x_field, RepeatFieldReference)
        y_is_repeat = isinstance(y_field, RepeatFieldReference)

        if x_is_repeat or y_is_repeat:
            return cls._repeat_chart_title(
                encoding,
                root=root,
                mark_type=mark_type,
                repeat_channel_def=y_def if y_is_repeat else x_def,
                dimension_channel_def=x_def if y_is_repeat else y_def,
            )

        if mark_type == "boxplot":
            measure = cls._boxplot_measure(encoding)
            measure_label = cls._field_display_label(cls._field_from_channel(measure), root=root)
            dimensions = cls._dimension_labels(encoding, root=root, exclude_defs=[measure])
            if measure_label:
                return cls._join_title(measure_label, "Distribution", dimensions)

        if mark_type in {"line", "area"}:
            if x_time and y_field:
                measure = cls._label_for_channel(y_def or {}, channel="y", root=root, mark_type=mark_type)
                time_label = cls._field_display_label(x_field, root=root, time_unit=x_time)
                groups = cls._group_labels(encoding, root=root, exclude_defs=[x_def, y_def])
                if measure and time_label:
                    return cls._join_title(measure, f"over {time_label}", groups)
            if y_time and x_field:
                measure = cls._label_for_channel(x_def or {}, channel="x", root=root, mark_type=mark_type)
                time_label = cls._field_display_label(y_field, root=root, time_unit=y_time)
                groups = cls._group_labels(encoding, root=root, exclude_defs=[x_def, y_def])
                if measure and time_label:
                    return cls._join_title(measure, f"over {time_label}", groups)

        if x_time and y_field:
            measure = cls._label_for_channel(y_def or {}, channel="y", root=root, mark_type=mark_type)
            time_label = cls._field_display_label(x_field, root=root, time_unit=x_time)
            groups = cls._group_labels(encoding, root=root, exclude_defs=[x_def, y_def])
            if measure and time_label:
                return cls._join_title(measure, f"over {time_label}", groups)

        if y_time and x_field:
            measure = cls._label_for_channel(x_def or {}, channel="x", root=root, mark_type=mark_type)
            time_label = cls._field_display_label(y_field, root=root, time_unit=y_time)
            groups = cls._group_labels(encoding, root=root, exclude_defs=[x_def, y_def])
            if measure and time_label:
                return cls._join_title(measure, f"over {time_label}", groups)

        if x_field and y_field:
            x_type = cls._channel_type(x_def)
            y_type = cls._channel_type(y_def)
            x_agg = cls._aggregate_from_channel(x_def)
            y_agg = cls._aggregate_from_channel(y_def)
            x_label = cls._label_for_channel(x_def or {}, channel="x", root=root, mark_type=mark_type)
            y_label = cls._label_for_channel(y_def or {}, channel="y", root=root, mark_type=mark_type)
            groups = cls._group_labels(encoding, root=root, exclude_defs=[x_def, y_def])

            if y_agg or (x_type in {"nominal", "ordinal"} and y_type == "quantitative"):
                return cls._join_title(y_label, f"by {x_label}" if x_label else "", groups)
            if x_agg or (y_type in {"nominal", "ordinal"} and x_type == "quantitative"):
                return cls._join_title(x_label, f"by {y_label}" if y_label else "", groups)
            if x_label and y_label:
                return cls._join_title(y_label, f"vs {x_label}", groups)

        measure = cls._primary_measure(encoding)
        dimension = cls._primary_dimension(encoding)
        if measure and dimension:
            measure_label = cls._label_for_channel(measure, channel="y", root=root, mark_type=mark_type)
            dimension_label = cls._field_display_label(cls._field_from_channel(dimension), root=root)
            if measure_label and dimension_label:
                return cls._join_title(measure_label, f"by {dimension_label}", [])
        return ""

    @staticmethod
    def _is_repeat_root(root: dict[str, Any]) -> bool:
        return isinstance(root.get("repeat"), (dict, list))

    @staticmethod
    def _mark_type(view: dict[str, Any]) -> str:
        mark = view.get("mark") if isinstance(view, dict) else None
        if isinstance(mark, str):
            return mark.strip().lower()
        if isinstance(mark, dict):
            value = mark.get("type")
            if isinstance(value, str):
                return value.strip().lower()
        return ""

    @classmethod
    def _repeat_chart_title(
            cls,
            encoding: dict[str, Any],
            *,
            root: dict[str, Any],
            mark_type: str,
            repeat_channel_def: dict[str, Any] | None,
            dimension_channel_def: dict[str, Any] | None,
    ) -> str:
        repeated_label = cls._repeated_measure_group_label(root)
        dimension_label = cls._field_display_label(cls._field_from_channel(dimension_channel_def), root=root)
        aggregate_label = cls._aggregate_label(cls._aggregate_from_channel(repeat_channel_def))

        if mark_type == "boxplot":
            distribution_subject = "Metric" if repeated_label == "Metrics" else "Repeated Measure"
            return cls._join_title(distribution_subject, "Distributions", [dimension_label])
        if aggregate_label:
            return cls._join_title(f"{aggregate_label} {repeated_label}", "", [dimension_label])
        return cls._join_title(repeated_label, "", [dimension_label])

    @classmethod
    def _boxplot_measure(cls, encoding: dict[str, Any]) -> dict[str, Any] | None:
        for channel in ("y", "x"):
            channel_def = cls._first_channel_def(encoding.get(channel))
            if isinstance(channel_def, dict) and cls._channel_type(channel_def) == "quantitative":
                return channel_def
        return cls._primary_measure(encoding)

    @classmethod
    def _dimension_labels(
            cls,
            encoding: dict[str, Any],
            *,
            root: dict[str, Any],
            exclude_defs: list[dict[str, Any] | None] | None = None,
    ) -> list[str]:
        labels: list[str] = []
        excluded = cls._field_identity_set(exclude_defs or [])
        for channel in ("x", "y", "color", "xOffset", "yOffset", "column", "row", "shape", "detail"):
            channel_def = cls._first_channel_def(encoding.get(channel))
            field = cls._field_from_channel(channel_def)
            if field is None:
                continue
            identity = cls._field_identity(field)
            if identity in excluded:
                continue
            if cls._channel_type(channel_def) == "quantitative" and not isinstance(field, RepeatFieldReference):
                continue
            label = cls._field_display_label(field, root=root, time_unit=cls._time_unit_from_channel(channel_def))
            if label:
                labels.append(label)
        return cls._unique_case_insensitive(labels)

    @classmethod
    def _group_labels(
            cls,
            encoding: dict[str, Any],
            *,
            root: dict[str, Any],
            exclude_defs: list[dict[str, Any] | None] | None = None,
    ) -> list[str]:
        labels: list[str] = []
        excluded = cls._field_identity_set(exclude_defs or [])
        for channel in ("color", "shape", "detail", "xOffset", "yOffset", "column", "row"):
            channel_def = cls._first_channel_def(encoding.get(channel))
            field = cls._field_from_channel(channel_def)
            if field is None:
                continue
            identity = cls._field_identity(field)
            if identity in excluded:
                continue
            label = cls._field_display_label(field, root=root, time_unit=cls._time_unit_from_channel(channel_def))
            if label:
                labels.append(label)
        return cls._unique_case_insensitive(labels)

    @classmethod
    def _join_title(cls, subject: str, relation: str, dimensions: list[str]) -> str:
        parts = [part.strip() for part in (subject, relation) if part and part.strip()]
        title = " ".join(parts).strip()
        unique_dimensions = cls._unique_case_insensitive([item for item in dimensions if item])
        if unique_dimensions:
            title = f"{title} by {cls._join_labels(unique_dimensions)}" if title else cls._join_labels(
                unique_dimensions)
        return re.sub(r"\s+", " ", title).strip()

    @staticmethod
    def _join_labels(labels: list[str]) -> str:
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]
        return f"{', '.join(labels[:-1])} and {labels[-1]}"

    @staticmethod
    def _field_identity(field: str | RepeatFieldReference | None) -> str:
        if field is None:
            return ""
        if isinstance(field, RepeatFieldReference):
            return f"repeat:{field.name.lower()}"
        return f"field:{str(field).strip().lower()}"

    @classmethod
    def _field_identity_set(cls, channel_defs: list[dict[str, Any] | None]) -> set[str]:
        return {cls._field_identity(cls._field_from_channel(item)) for item in channel_defs if isinstance(item, dict)}

    @staticmethod
    def _unique_case_insensitive(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = value.strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                result.append(text)
        return result

    @classmethod
    def _remove_redundant_color_legend(cls, encoding: dict[str, Any], *, changes: list[str]) -> None:
        color = cls._first_channel_def(encoding.get("color"))
        if not isinstance(color, dict):
            return
        color_field = cls._field_identity(cls._field_from_channel(color))
        if not color_field:
            return
        for channel in ("x", "y"):
            channel_def = cls._first_channel_def(encoding.get(channel))
            if color_field == cls._field_identity(cls._field_from_channel(channel_def)):
                if color.get("legend") is not None:
                    color["legend"] = None
                    changes.append(f"Removed redundant color legend because color duplicates {channel} field.")
                return

    @staticmethod
    def _remove_channel_title(channel_def: dict[str, Any], *, changes: list[str], owner: str) -> None:
        if "title" in channel_def:
            old_title = channel_def.pop("title")
            if old_title:
                changes.append(f"Removed conflicting {owner}.title {old_title!r}; nested title is authoritative.")

    @classmethod
    def _primary_measure(cls, encoding: dict[str, Any]) -> dict[str, Any] | None:
        for channel in ("y", "x", "theta", "size", "radius"):
            channel_def = cls._first_channel_def(encoding.get(channel))
            if isinstance(channel_def, dict) and cls._channel_type(channel_def) == "quantitative":
                return channel_def
        return None

    @classmethod
    def _primary_dimension(cls, encoding: dict[str, Any]) -> dict[str, Any] | None:
        for channel in ("x", "y", "color", "shape", "column", "row"):
            channel_def = cls._first_channel_def(encoding.get(channel))
            if isinstance(channel_def, dict) and cls._channel_type(channel_def) in {"nominal", "ordinal", "temporal"}:
                return channel_def
        return None

    @classmethod
    def _label_for_channel(
            cls,
            channel_def: dict[str, Any],
            *,
            channel: str,
            root: dict[str, Any],
            mark_type: str = "",
    ) -> str:
        field = cls._field_from_channel(channel_def)
        if field is None:
            aggregate = cls._aggregate_from_channel(channel_def)
            return cls._aggregate_label(aggregate) if aggregate else ""
        time_unit = cls._time_unit_from_channel(channel_def)
        aggregate = cls._aggregate_from_channel(channel_def)
        if mark_type == "boxplot" and channel == cls.TOOLTIP_CHANNEL:
            aggregate = ""
        return cls._measure_label(field, aggregate=aggregate, root=root, time_unit=time_unit)

    @classmethod
    def _measure_label(
            cls,
            field: str | RepeatFieldReference,
            *,
            aggregate: str = "",
            root: dict[str, Any],
            time_unit: str = "",
    ) -> str:
        field_label = cls._field_display_label(field, root=root, time_unit=time_unit)
        aggregate_label = cls._aggregate_label(aggregate)
        if not aggregate_label:
            return field_label
        if aggregate_label == "Count" and field_label in {"*", "Record", "Records"}:
            return "Count of Records"
        if isinstance(field, RepeatFieldReference):
            if aggregate_label == "Count":
                return "Count"
            return f"{aggregate_label} Value"
        return f"{aggregate_label} {field_label}".strip()

    @classmethod
    def _field_display_label(
            cls,
            field: str | RepeatFieldReference | None,
            *,
            root: dict[str, Any],
            time_unit: str = "",
    ) -> str:
        if field is None:
            return ""
        if isinstance(field, RepeatFieldReference):
            return "Value"
        if field == "*":
            return "Records"
        if time_unit:
            label = cls.TIME_UNIT_LABELS.get(time_unit.lower())
            if label:
                return label
        return cls._humanize_field_name(field)

    @classmethod
    def _repeated_measure_group_label(cls, root: dict[str, Any]) -> str:
        repeat_fields = cls._all_repeat_fields(root)
        if not repeat_fields:
            return "Repeated Measures"
        if cls._fields_look_like_metrics(repeat_fields):
            return "Metrics"
        return "Repeated Measures"

    @staticmethod
    def _all_repeat_fields(root: dict[str, Any]) -> list[str]:
        repeat = root.get("repeat")
        fields: list[str] = []
        if isinstance(repeat, dict):
            for key in ("row", "column", "layer"):
                value = repeat.get(key)
                if isinstance(value, list):
                    fields.extend(str(item) for item in value if isinstance(item, str) and item.strip())
                elif isinstance(value, str) and value.strip():
                    fields.append(value.strip())
        elif isinstance(repeat, list):
            fields.extend(str(item) for item in repeat if isinstance(item, str) and item.strip())
        return fields

    @staticmethod
    def _fields_look_like_metrics(fields: list[str]) -> bool:
        if len(fields) < 2:
            return False
        metric_tokens = {
            "score", "rate", "ratio", "value", "metric", "measure", "psnr", "ssim", "lpips",
            "rmse", "mae", "mse", "accuracy", "precision", "recall", "runtime", "memory", "params",
            "sales", "profit", "revenue", "cost", "count", "total", "mean", "average",
        }
        normalized = " ".join(fields).replace("_", " ").replace("-", " ").lower()
        return any(token in normalized for token in metric_tokens)

    @staticmethod
    def _field_from_channel(channel_def: dict[str, Any] | None) -> str | RepeatFieldReference | None:
        if not isinstance(channel_def, dict):
            return None
        field = channel_def.get("field")
        if isinstance(field, str) and field.strip():
            return field.strip()
        if isinstance(field, dict):
            repeat = field.get("repeat")
            if isinstance(repeat, str) and repeat.strip():
                return RepeatFieldReference(repeat.strip())
        return None

    @staticmethod
    def _first_channel_def(channel_def: Any) -> dict[str, Any] | None:
        if isinstance(channel_def, dict):
            return channel_def
        if isinstance(channel_def, list):
            for item in channel_def:
                if isinstance(item, dict):
                    return item
        return None

    @staticmethod
    def _channel_type(channel_def: dict[str, Any] | None) -> str:
        if not isinstance(channel_def, dict):
            return ""
        value = channel_def.get("type")
        return str(value or "").strip().lower()

    @staticmethod
    def _aggregate_from_channel(channel_def: dict[str, Any] | None) -> str:
        if not isinstance(channel_def, dict):
            return ""
        aggregate = channel_def.get("aggregate")
        if isinstance(aggregate, str):
            return aggregate.strip().lower()
        return ""

    @staticmethod
    def _time_unit_from_channel(channel_def: dict[str, Any] | None) -> str:
        if not isinstance(channel_def, dict):
            return ""
        time_unit = channel_def.get("timeUnit")
        if isinstance(time_unit, str):
            return time_unit.strip().lower()
        return ""

    @classmethod
    def _aggregate_label(cls, aggregate: str) -> str:
        if not aggregate:
            return ""
        return cls.AGGREGATE_LABELS.get(aggregate.strip().lower(), cls._humanize_field_name(aggregate))

    @classmethod
    def _canonical_aggregate(cls, aggregate: str) -> str:
        return cls.AGGREGATE_WORDS.get(aggregate.strip().lower(), aggregate.strip().lower())

    @staticmethod
    def _humanize_field_name(field: str) -> str:
        text = str(field).strip()
        if not text:
            return ""
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        words = [word for word in text.split() if word]
        if not words:
            return ""
        result: list[str] = []
        for word in words:
            if word.isupper() or any(char.isdigit() for char in word):
                result.append(word)
            elif len(word) <= 4 and word.lower() in {"psnr", "ssim", "lpips", "rmse", "mae", "mse"}:
                result.append(word.upper())
            else:
                result.append(word[:1].upper() + word[1:])
        return " ".join(result)

    @staticmethod
    def _set_nested_title(container: dict[str, Any], label: str, *, changes: list[str], owner: str) -> None:
        current = container.get("title")
        if current == label:
            return
        container["title"] = label
        changes.append(f"Set {owner}.title to {label!r}.")

    @classmethod
    def _set_channel_title(cls, channel_def: dict[str, Any], label: str, *, changes: list[str], channel: str) -> None:
        current = channel_def.get("title")
        if current == label:
            return
        channel_def["title"] = label
        changes.append(f"Set encoding.{channel}.title to {label!r}.")

    @classmethod
    def _set_view_title(cls, view: dict[str, Any], title: str, *, changes: list[str]) -> None:
        current = cls._title_text(view.get("title"))
        if current == title:
            return
        if isinstance(view.get("title"), dict):
            view["title"]["text"] = title
        else:
            view["title"] = title
        changes.append(f"Set chart title to {title!r}.")

    @staticmethod
    def _title_text(title: Any) -> str:
        if isinstance(title, str):
            return title.strip()
        if isinstance(title, dict):
            text = title.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def _collect_label_uses(
            cls,
            node: Any,
            label_uses: dict[PresentationLabelKey, list[PresentationLabelUse]],
            *,
            path: tuple[Any, ...],
    ) -> None:
        if isinstance(node, dict):
            encoding = node.get("encoding")
            if isinstance(encoding, dict):
                for channel, channel_def in encoding.items():
                    cls._collect_channel_label_uses(
                        channel=channel,
                        channel_def=channel_def,
                        label_uses=label_uses,
                        path=(*path, "encoding", channel),
                    )
            for key, value in node.items():
                if key == "encoding":
                    continue
                cls._collect_label_uses(value, label_uses, path=(*path, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                cls._collect_label_uses(item, label_uses, path=(*path, index))

    @classmethod
    def _collect_channel_label_uses(
            cls,
            *,
            channel: str,
            channel_def: Any,
            label_uses: dict[PresentationLabelKey, list[PresentationLabelUse]],
            path: tuple[Any, ...],
    ) -> None:
        if isinstance(channel_def, list):
            for index, item in enumerate(channel_def):
                cls._collect_channel_label_uses(
                    channel=channel,
                    channel_def=item,
                    label_uses=label_uses,
                    path=(*path, index),
                )
            return
        if not isinstance(channel_def, dict):
            return
        key = cls._label_key(channel_def)
        if key is None:
            return
        priority_base = cls.CHANNEL_PRIORITY.get(channel, 50)
        labels = cls._labels_from_channel_def(channel_def)
        for container_kind, label in labels:
            label_uses.setdefault(key, []).append(
                PresentationLabelUse(
                    label=label,
                    priority=priority_base + cls._container_priority(container_kind),
                    path=path,
                    container_kind=container_kind,
                )
            )

    @staticmethod
    def _label_key(channel_def: dict[str, Any]) -> PresentationLabelKey | None:
        field = channel_def.get("field")
        if not isinstance(field, str) or not field.strip():
            return None
        aggregate = channel_def.get("aggregate")
        time_unit = channel_def.get("timeUnit")
        bin_value = channel_def.get("bin")
        return PresentationLabelKey(
            field=field.strip(),
            aggregate=str(aggregate or "").strip().lower(),
            bin=str(bin_value or "").strip().lower(),
            time_unit=str(time_unit or "").strip().lower(),
        )

    @staticmethod
    def _labels_from_channel_def(channel_def: dict[str, Any]) -> list[tuple[str, str]]:
        labels: list[tuple[str, str]] = []
        title = channel_def.get("title")
        if isinstance(title, str) and title.strip():
            labels.append(("channel_title", title.strip()))
        axis = channel_def.get("axis")
        if isinstance(axis, dict):
            axis_title = axis.get("title")
            if isinstance(axis_title, str) and axis_title.strip():
                labels.append(("axis_title", axis_title.strip()))
        legend = channel_def.get("legend")
        if isinstance(legend, dict):
            legend_title = legend.get("title")
            if isinstance(legend_title, str) and legend_title.strip():
                labels.append(("legend_title", legend_title.strip()))
        return labels

    @staticmethod
    def _container_priority(container_kind: str) -> int:
        return {
            "axis_title": 0,
            "legend_title": 5,
            "channel_title": 10,
        }.get(container_kind, 20)

    @staticmethod
    def _choose_canonical_label(uses: list[PresentationLabelUse]) -> str:
        ordered = sorted(uses, key=lambda item: (item.priority, len(item.label), item.label.lower()))
        return ordered[0].label.strip() if ordered else ""

    @staticmethod
    def _set_label(spec: dict[str, Any], path: tuple[Any, ...], label: str, container_kind: str) -> None:
        node: Any = spec
        for part in path:
            node = node[part]
        if not isinstance(node, dict):
            return
        if container_kind == "axis_title":
            axis = node.setdefault("axis", {})
            if isinstance(axis, dict):
                axis["title"] = label
            return
        if container_kind == "legend_title":
            legend = node.setdefault("legend", {})
            if isinstance(legend, dict):
                legend["title"] = label
            return
        node["title"] = label

    @classmethod
    def _normalize_titles(cls, node: Any, replacements: dict[str, str]) -> None:
        if isinstance(node, dict):
            title = node.get("title")
            if isinstance(title, str) and title.strip():
                node["title"] = cls._replace_labels(title, replacements)
            elif isinstance(title, dict):
                text = title.get("text")
                if isinstance(text, str):
                    title["text"] = cls._replace_labels(text, replacements)
            for value in node.values():
                cls._normalize_titles(value, replacements)
        elif isinstance(node, list):
            for item in node:
                cls._normalize_titles(item, replacements)

    @staticmethod
    def _replace_labels(text: str, replacements: dict[str, str]) -> str:
        result = text
        for source in sorted(replacements, key=len, reverse=True):
            target = replacements[source]
            if source and source != target:
                result = result.replace(source, target)
        return result

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = value.strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result
