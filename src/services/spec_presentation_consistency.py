from __future__ import annotations

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


class SpecPresentationConsistencyService:
    """Conservative presentation-layer normalization for Vega-Lite specs.

    This service does not try to infer semantic synonyms. It only enforces one rule:
    the same encoded field/aggregate/bin/timeUnit within one specification should use
    one visible label across axis, legend, tooltip and explicit channel titles.
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

    def normalize(self, spec: dict[str, Any]) -> PresentationConsistencyResult:
        if not isinstance(spec, dict):
            return PresentationConsistencyResult(spec={})
        normalized = deepcopy(spec)
        label_uses: dict[PresentationLabelKey, list[PresentationLabelUse]] = {}
        self._collect_label_uses(normalized, label_uses, path=())
        changes: list[str] = []
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
