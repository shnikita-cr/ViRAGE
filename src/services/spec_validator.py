from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.domain.models import SpecValidationResult, VegaLiteSpecArtifact
from src.services.base import BaseService

_ALLOWED_MARKS = {"line", "area", "bar", "point", "circle", "boxplot", "histogram", "tick"}
_ALLOWED_CHANNELS = {"x", "y", "color", "tooltip", "detail"}
_ALLOWED_TYPES = {"quantitative", "temporal", "nominal", "ordinal"}
_ALLOWED_AGGREGATES = {"mean", "sum", "count", "min", "max", "median"}
_ALLOWED_TRANSFORMS = {"aggregate", "filter", "calculate", "bin"}


class SpecValidatorService(BaseService):
    def invoke(self, vega_spec: VegaLiteSpecArtifact) -> SpecValidationResult:
        spec = dict(vega_spec.spec_json)
        errors: list[str] = []
        repair_hints: list[str] = []

        if not isinstance(spec, dict):
            return SpecValidationResult(
                validated_spec={},
                validation_errors=["Specification must be a JSON object."],
                repair_hints=["Return a Vega-Lite JSON object, not free-form text."],
                is_valid=False,
            )

        self._validate_required_keys(spec, errors)
        dataset_columns = self._load_columns(spec, errors)
        mark_type = self._validate_mark(spec, errors)
        encoding = self._validate_encoding(spec, dataset_columns, errors)
        self._validate_transforms(spec, dataset_columns, errors)
        self._validate_mark_specific_requirements(mark_type, encoding, errors)

        if errors:
            repair_hints.extend([
                "Ensure the spec contains $schema, data.url, mark and encoding.",
                "Use only columns that exist in the prepared dataset.",
                f"Restrict mark.type to supported values: {sorted(_ALLOWED_MARKS)}.",
                "Keep encodings explicit and use valid Vega-Lite field types.",
                "Use only simple transforms: aggregate, filter, calculate, bin.",
            ])
            return SpecValidationResult(validated_spec={}, validation_errors=errors, repair_hints=repair_hints, is_valid=False)

        normalized = self._normalize_spec(spec)
        return SpecValidationResult(validated_spec=normalized, validation_errors=[], repair_hints=[], is_valid=True)

    @staticmethod
    def _validate_required_keys(spec: dict[str, Any], errors: list[str]) -> None:
        for key in ["$schema", "data", "mark", "encoding"]:
            if key not in spec:
                errors.append(f"Missing required key: {key}.")

    @staticmethod
    def _load_columns(spec: dict[str, Any], errors: list[str]) -> set[str]:
        data = spec.get("data", {})
        data_url = data.get("url") if isinstance(data, dict) else None
        if not isinstance(data_url, str) or not data_url.strip():
            errors.append("Specification data.url must point to the prepared dataset path.")
            return set()
        path = Path(data_url)
        if not path.exists():
            errors.append(f"Prepared dataset path does not exist: {data_url}")
            return set()
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception as exc:
            errors.append(f"Prepared dataset could not be read: {exc}")
            return set()
        return set(df.columns)

    @staticmethod
    def _validate_mark(spec: dict[str, Any], errors: list[str]) -> str:
        mark = spec.get("mark")
        mark_type = mark.get("type") if isinstance(mark, dict) else mark
        if not isinstance(mark_type, str) or not mark_type.strip():
            errors.append("Specification mark must be a non-empty string or a mark object with a type.")
            return ""
        if mark_type not in _ALLOWED_MARKS:
            errors.append(f"Unsupported mark type: {mark_type}.")
        return mark_type

    @staticmethod
    def _validate_encoding(spec: dict[str, Any], dataset_columns: set[str], errors: list[str]) -> dict[str, dict[str, Any]]:
        encoding = spec.get("encoding", {})
        if not isinstance(encoding, dict) or not encoding:
            errors.append("Specification encoding must be a non-empty object.")
            return {}
        for channel, channel_spec in encoding.items():
            if channel not in _ALLOWED_CHANNELS:
                errors.append(f"Unsupported encoding channel: {channel}.")
                continue
            if not isinstance(channel_spec, dict):
                errors.append(f"Encoding for channel '{channel}' must be an object.")
                continue
            field = channel_spec.get("field")
            if channel != "detail" and channel != "tooltip":
                if not isinstance(field, str) or not field.strip():
                    errors.append(f"Encoding.{channel}.field is required.")
                elif dataset_columns and field not in dataset_columns and field != "count":
                    errors.append(f"Encoding.{channel}.field references a missing dataset column: {field}.")
            field_type = channel_spec.get("type")
            if field_type is not None and field_type not in _ALLOWED_TYPES:
                errors.append(f"Encoding.{channel}.type has unsupported value: {field_type}.")
            aggregate = channel_spec.get("aggregate")
            if aggregate is not None and aggregate not in _ALLOWED_AGGREGATES:
                errors.append(f"Encoding.{channel}.aggregate has unsupported value: {aggregate}.")
        return encoding

    @staticmethod
    def _validate_transforms(spec: dict[str, Any], dataset_columns: set[str], errors: list[str]) -> None:
        transforms = spec.get("transform", [])
        if transforms is None:
            return
        if not isinstance(transforms, list):
            errors.append("Specification transform must be a list when present.")
            return
        for index, transform in enumerate(transforms):
            if not isinstance(transform, dict):
                errors.append(f"Transform at index {index} must be an object.")
                continue
            kind = transform.get("kind")
            if kind is None:
                # Support plain Vega-Lite transforms when produced directly by model.
                if "aggregate" in transform or "joinaggregate" in transform:
                    kind = "aggregate"
                elif "filter" in transform:
                    kind = "filter"
                elif "calculate" in transform:
                    kind = "calculate"
                elif "bin" in transform:
                    kind = "bin"
            if kind not in _ALLOWED_TRANSFORMS:
                errors.append(f"Unsupported transform kind at index {index}: {kind}.")
                continue
            field = transform.get("field") or transform.get("field_name")
            if field and dataset_columns and field not in dataset_columns and field != "count":
                errors.append(f"Transform at index {index} references a missing field: {field}.")
            aggregate = transform.get("aggregate")
            if aggregate is not None and aggregate not in _ALLOWED_AGGREGATES:
                errors.append(f"Transform at index {index} uses unsupported aggregate: {aggregate}.")

    @staticmethod
    def _validate_mark_specific_requirements(mark_type: str, encoding: dict[str, dict[str, Any]], errors: list[str]) -> None:
        x = encoding.get("x") if isinstance(encoding, dict) else None
        y = encoding.get("y") if isinstance(encoding, dict) else None
        if mark_type in {"line", "area", "bar", "point", "circle", "tick"}:
            if not isinstance(x, dict) or not x.get("field"):
                errors.append(f"Mark '{mark_type}' requires encoding.x.field.")
            if not isinstance(y, dict) or not y.get("field"):
                errors.append(f"Mark '{mark_type}' requires encoding.y.field.")
        if mark_type == "boxplot" and not isinstance(y, dict):
            errors.append("Boxplot requires a quantitative y encoding.")
        if mark_type == "histogram":
            if not isinstance(x, dict) or not x.get("field"):
                errors.append("Histogram requires encoding.x.field for the binned measure.")
            if x and x.get("type") not in {None, "quantitative"}:
                errors.append("Histogram encoding.x.type must be quantitative.")

    @staticmethod
    def _normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(spec)
        normalized.setdefault("$schema", "https://vega.github.io/schema/vega-lite/v5.json")
        if isinstance(normalized.get("mark"), str):
            normalized["mark"] = normalized["mark"].strip()
        return normalized
