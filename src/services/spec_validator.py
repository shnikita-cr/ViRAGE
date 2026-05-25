from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.domain.models import SpecValidationResult, VegaLiteSpecArtifact
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.data import read_dataframe
from src.services.spec_repair import SpecRepairService

VEGA_LITE_SCHEMA_URL = 'https://vega.github.io/schema/vega-lite/v5.json'
_COMPOSITION_KEYS = {'layer', 'facet', 'repeat', 'concat', 'hconcat', 'vconcat'}


class SpecValidatorService(BaseService):
    """Validate generated Vega-Lite specs without enforcing a ViRAGE subset policy.

    Validation is intentionally split into two levels:
    1. Vega-Lite technical validation: the runtime can compile/render the spec and it is not empty.
    2. Project data validation: the attached data can be read and referenced fields exist in the prepared table
       or are produced by Vega-Lite transforms.

    The validator does not reject Vega-Lite mark types, composition types, encoding channels, or transform kinds just
    because they are not explicitly enumerated in ViRAGE. This allows the spec generator to use the full Vega-Lite
    grammar supported by the Vega-Lite runtime.
    """

    def invoke(self, vega_spec: VegaLiteSpecArtifact, runtime: RuntimeContext | None = None) -> SpecValidationResult:
        spec = deepcopy(vega_spec.spec_json)
        errors: list[str] = []
        repair_hints: list[str] = []

        if not isinstance(spec, dict):
            return SpecValidationResult(
                validated_spec={},
                validation_errors=['Specification must be a JSON object.'],
                repair_hints=['Return one Vega-Lite JSON object, not free-form text.'],
                is_valid=False,
            )

        normalized, repair_notes = self._normalize_spec(spec)
        repair_hints.extend(repair_notes)
        dataset_path, dataset_columns = self._load_runtime_data(normalized, errors, runtime=runtime)
        self._validate_vega_lite_shape(normalized, errors)

        if dataset_path is not None and dataset_columns:
            self._validate_referenced_fields(normalized, dataset_columns, errors, repair_hints)
            validity = self._validate_with_vega_runtime(normalized, dataset_path, runtime=runtime)
            repair_hints.extend(validity['repair_hints'])

            # Technical failure: if the Vega runtime cannot compile/render, the graph-level technical loop should retry.
            if not validity['is_valid_scenegraph']:
                errors.append(validity['scenegraph_error'] or 'Vega-Lite runtime could not compile the specification.')
            if validity['is_empty_scenegraph']:
                errors.append('Validated specification renders an empty scenegraph.')
                repair_hints.append(
                    'Change fields, filters, transforms, or chart type so at least one visible mark is rendered.')

            # Schema validation is logged as a hint unless rendering also fails. Vega-Lite/Vega can render some specs that
            # Altair rejects because of wrapper limitations; these should not be treated as ViRAGE subset failures.
            if not validity['is_valid_schema'] and validity['schema_error']:
                repair_hints.append(f'Altair schema warning: {validity["schema_error"]}')

        if errors:
            repair_hints.extend([
                'Return a complete Vega-Lite v5 specification with $schema and runtime data.url.',
                'Use fields that exist in the prepared dataset or are created with transform.as.',
                'If you use layer/facet/repeat/concat, ensure every child view is a valid Vega-Lite specification.',
                'Do not rely on unsupported JavaScript expressions or malformed transform objects.',
            ])
            return SpecValidationResult(
                validated_spec=normalized,
                validation_errors=self._dedupe(errors),
                repair_hints=self._dedupe(repair_hints),
                is_valid=False,
            )

        return SpecValidationResult(
            validated_spec=normalized,
            validation_errors=[],
            repair_hints=self._dedupe(repair_hints),
            is_valid=True,
        )

    @staticmethod
    def _normalize_spec(spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        return SpecRepairService().repair(spec)

    @staticmethod
    def _validate_vega_lite_shape(spec: dict[str, Any], errors: list[str]) -> None:
        if '$schema' not in spec:
            errors.append('Missing required key: $schema.')
        if not SpecValidatorService._has_any_view_spec(spec):
            errors.append(
                'Specification must contain a Vega-Lite view: mark/encoding or one of layer/facet/repeat/concat/hconcat/vconcat.')

    @staticmethod
    def _has_any_view_spec(node: Any) -> bool:
        if isinstance(node, dict):
            if 'mark' in node or _COMPOSITION_KEYS.intersection(node.keys()):
                return True
            return any(SpecValidatorService._has_any_view_spec(value) for value in node.values())
        if isinstance(node, list):
            return any(SpecValidatorService._has_any_view_spec(item) for item in node)
        return False

    @staticmethod
    def _load_runtime_data(spec: dict[str, Any], errors: list[str], *, runtime: RuntimeContext | None = None) -> tuple[Path | None, set[str]]:
        data = spec.get('data', {})
        data_url = data.get('url') if isinstance(data, dict) else None
        if not isinstance(data_url, str) or not data_url.strip():
            errors.append('Specification data.url must point to the prepared dataset path.')
            return None, set()
        try:
            df = runtime.read_dataframe(data_url, nrows=5) if runtime is not None else read_dataframe(data_url, nrows=5)
        except Exception as exc:
            errors.append(f'Prepared dataset could not be read: {exc}')
            return None, set()
        return Path(data_url), set(str(column) for column in df.columns)

    @classmethod
    def _validate_referenced_fields(
            cls,
            spec: dict[str, Any],
            dataset_columns: set[str],
            errors: list[str],
            repair_hints: list[str],
    ) -> None:
        derived_fields = cls._derived_fields_from_spec(spec)
        allowed_fields = dataset_columns | derived_fields
        referenced_fields = cls._field_references_from_spec(spec)
        missing = sorted(field for field in referenced_fields if field not in allowed_fields)
        if missing:
            errors.append(
                'Specification references fields not present in prepared data or transform outputs: '
                + ', '.join(missing)
            )
            repair_hints.append(
                'Use safe prepared-data column names only, or create derived fields with transform.as before referencing them.'
            )

    @classmethod
    def _derived_fields_from_spec(cls, spec: dict[str, Any]) -> set[str]:
        fields: set[str] = set()
        for node in cls._walk(spec):
            if not isinstance(node, dict):
                continue
            if 'as' in node:
                cls._add_as_value(fields, node.get('as'))
            for transform_key in ('aggregate', 'joinaggregate', 'window'):
                value = node.get(transform_key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            cls._add_as_value(fields, item.get('as'))
        return fields

    @classmethod
    def _field_references_from_spec(cls, spec: dict[str, Any]) -> set[str]:
        fields: set[str] = set()
        for node in cls._walk(spec):
            if not isinstance(node, dict):
                continue
            value = node.get('field')
            if isinstance(value, str) and value.strip() and value.strip() != '*':
                fields.add(value.strip())
            # Some transforms use field lists without the literal key "field".
            for key in ('fields', 'groupby', 'sort'):
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            fields.add(item.strip())
            for key in ('key', 'from', 'lookup'):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    fields.add(value.strip())
        return fields

    @staticmethod
    def _add_as_value(fields: set[str], value: Any) -> None:
        if isinstance(value, str) and value.strip():
            fields.add(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    fields.add(item.strip())

    @staticmethod
    def _walk(node: Any) -> Iterable[Any]:
        yield node
        if isinstance(node, dict):
            for value in node.values():
                yield from SpecValidatorService._walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from SpecValidatorService._walk(item)

    @classmethod
    def _validate_with_vega_runtime(cls, spec: dict[str, Any], dataset_path: Path, *, runtime: RuntimeContext | None = None) -> dict[str, Any]:
        repair_hints: list[str] = []
        try:
            import vl_convert as vlc  # type: ignore
        except ImportError as exc:
            repair_hints.append(f'Vega-Lite runtime validation skipped because vl-convert-python is unavailable: {exc}')
            return {
                'is_valid_schema': False,
                'is_valid_scenegraph': True,
                'is_empty_scenegraph': False,
                'schema_error': None,
                'scenegraph_error': None,
                'repair_hints': repair_hints,
            }

        try:
            df = runtime.read_dataframe(dataset_path) if runtime is not None else read_dataframe(dataset_path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return {
                'is_valid_schema': False,
                'is_valid_scenegraph': False,
                'is_empty_scenegraph': True,
                'schema_error': f'Could not read dataset: {exc}',
                'scenegraph_error': f'Could not read dataset: {exc}',
                'repair_hints': repair_hints,
            }

        schema_error = None
        scenegraph_error = None
        is_valid_schema = False
        is_valid_scenegraph = False
        is_empty_scenegraph = True

        try:
            import altair as alt
            alt.Chart.from_dict(cls._spec_add_data(spec, df.head()))
            is_valid_schema = True
        except ImportError as exc:
            schema_error = f'Altair is unavailable; schema validation skipped: {exc}'
            repair_hints.append(schema_error)
        except Exception as exc:
            schema_error = str(exc)

        try:
            scenegraph = vlc.vegalite_to_scenegraph(vl_spec=cls._spec_add_data(spec, df), show_warnings=False)
            is_valid_scenegraph = True
            is_empty_scenegraph = cls._is_chart_empty_scenegraph(scenegraph)
        except Exception as exc:
            scenegraph_error = str(exc)

        return {
            'is_valid_schema': is_valid_schema,
            'is_valid_scenegraph': is_valid_scenegraph,
            'is_empty_scenegraph': is_empty_scenegraph,
            'schema_error': schema_error,
            'scenegraph_error': scenegraph_error,
            'repair_hints': repair_hints,
        }

    @staticmethod
    def _spec_add_data(spec: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        clone = deepcopy(spec)
        clone['data'] = {'values': df.where(pd.notna(df), None).to_dict(orient='records')}
        return clone

    @staticmethod
    def _get_scenegraph_field(scenegraph: dict, key: str, value: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if node.get(key) == value:
                    results.append(node)
                for item in node.values():
                    walk(item)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(scenegraph)
        return results

    @classmethod
    def _is_chart_empty_scenegraph(cls, scenegraph: dict) -> bool:
        def is_mark_group_empty(mark_role: dict[str, Any]) -> bool:
            if 'items' not in mark_role or len(mark_role['items']) == 0:
                return True
            marktype = mark_role.get('marktype')
            items = mark_role['items']
            if marktype in ('line', 'area', 'trail'):
                return len(items) <= 1 or all(not item.get('defined', True) for item in items)
            if marktype == 'symbol':
                return all(item.get('size', 42) == 0 or len(item.get('shape', 'circle')) == 0 for item in items)
            if marktype == 'shape':
                return all(len(item.get('shape', 'geoshape')) == 0 for item in items)
            if marktype == 'rect':
                return all(item.get('height', 1) == 0 or item.get('width', 1) == 0 for item in items)
            return False

        mark_roles = cls._get_scenegraph_field(scenegraph, 'role', 'mark')
        if not mark_roles:
            return True
        return all(is_mark_group_empty(mark_role) for mark_role in mark_roles)

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = str(value).strip()
            if key and key not in seen:
                seen.add(key)
                result.append(key)
        return result
