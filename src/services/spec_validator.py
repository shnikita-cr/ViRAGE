from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.domain.models import SpecValidationResult, VegaLiteSpecArtifact
from src.services.base import BaseService
from src.services.data import read_dataframe

_ALLOWED_MARKS = {'line', 'area', 'bar', 'point', 'circle', 'square', 'boxplot', 'histogram', 'tick', 'rect', 'rule',
                  'text'}
_ALLOWED_CHANNELS = {'x', 'y', 'color', 'tooltip', 'detail', 'size', 'shape', 'opacity', 'row', 'column', 'theta',
                     'radius'}
_ALLOWED_TYPES = {'quantitative', 'temporal', 'nominal', 'ordinal', 'geojson'}
_ALLOWED_AGGREGATES = {'mean', 'average', 'sum', 'count', 'min', 'max', 'median'}
_ALLOWED_TRANSFORMS = {'aggregate', 'joinaggregate', 'filter', 'calculate', 'bin', 'timeUnit', 'window'}
_MARK_REQUIRING_XY = {'line', 'area', 'bar', 'point', 'circle', 'square', 'tick', 'rect', 'rule'}


class SpecValidatorService(BaseService):
    def invoke(self, vega_spec: VegaLiteSpecArtifact) -> SpecValidationResult:
        spec = deepcopy(vega_spec.spec_json)
        errors: list[str] = []
        repair_hints: list[str] = []

        if not isinstance(spec, dict):
            return SpecValidationResult(validated_spec={}, validation_errors=['Specification must be a JSON object.'],
                                        repair_hints=['Return a Vega-Lite JSON object, not free-form text.'],
                                        is_valid=False)

        self._validate_required_keys(spec, errors)
        dataset_path, dataset_columns = self._load_columns(spec, errors)
        mark_type = self._validate_mark(spec, errors)
        encoding = self._validate_encoding(spec, dataset_columns, errors)
        self._validate_transforms(spec, dataset_columns, errors)
        self._validate_mark_specific_requirements(mark_type, encoding, errors)

        normalized = self._normalize_spec(spec)
        if dataset_path is not None and not errors:
            validity = self._validate_with_vega_runtime(normalized, dataset_path)
            if not validity['is_valid_scenegraph']:
                errors.append(validity['scenegraph_error'] or 'Scenegraph validation failed.')
            if not validity['is_valid_schema']:
                errors.append(validity['schema_error'] or 'Schema validation failed.')
            if validity['is_empty_scenegraph']:
                errors.append('Validated specification renders an empty scenegraph.')
                repair_hints.append('Adjust encoding, filters or transforms so at least one visual mark is rendered.')

        if errors:
            repair_hints.extend([
                'Ensure the spec contains $schema, data.url, mark and encoding.',
                'Use only columns that exist in the prepared dataset or fields derived by transform.as.',
                f'Restrict mark.type to supported values: {sorted(_ALLOWED_MARKS)}; use point for scatter plots.',
                'Keep encodings explicit and use valid Vega-Lite field types.',
                'Use Vega-Lite transforms with standard keys such as aggregate/filter/calculate/bin/timeUnit.',
            ])
            return SpecValidationResult(validated_spec=normalized, validation_errors=self._dedupe(errors),
                                        repair_hints=self._dedupe(repair_hints), is_valid=False)

        return SpecValidationResult(validated_spec=normalized, validation_errors=[], repair_hints=[], is_valid=True)

    @staticmethod
    def _validate_required_keys(spec: dict[str, Any], errors: list[str]) -> None:
        for key in ['$schema', 'data', 'mark', 'encoding']:
            if key not in spec:
                errors.append(f'Missing required key: {key}.')

    @staticmethod
    def _load_columns(spec: dict[str, Any], errors: list[str]) -> tuple[Path | None, set[str]]:
        data = spec.get('data', {})
        data_url = data.get('url') if isinstance(data, dict) else None
        if not isinstance(data_url, str) or not data_url.strip():
            errors.append('Specification data.url must point to the prepared dataset path.')
            return None, set()
        path = data_url
        try:
            df = read_dataframe(path, nrows=5)
        except Exception as exc:
            errors.append(f'Prepared dataset could not be read: {exc}')
            return None, set()
        return Path(data_url), set(df.columns)

    @staticmethod
    def _validate_mark(spec: dict[str, Any], errors: list[str]) -> str:
        mark = spec.get('mark')
        mark_type = mark.get('type') if isinstance(mark, dict) else mark
        if isinstance(mark_type, str) and mark_type.strip().lower() == 'scatter':
            mark_type = 'point'
            if isinstance(mark, dict):
                mark['type'] = 'point'
            else:
                spec['mark'] = 'point'
        if not isinstance(mark_type, str) or not mark_type.strip():
            errors.append('Specification mark must be a non-empty string or a mark object with a type.')
            return ''
        mark_type = mark_type.strip()
        if mark_type not in _ALLOWED_MARKS:
            errors.append(f'Unsupported mark type: {mark_type}.')
        return mark_type

    @staticmethod
    def _derived_fields_from_transforms(spec: dict[str, Any]) -> set[str]:
        fields: set[str] = set()
        transforms = spec.get('transform', [])
        if not isinstance(transforms, list):
            return fields
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            as_value = transform.get('as')
            if isinstance(as_value, str):
                fields.add(as_value)
            elif isinstance(as_value, list):
                fields.update(str(item) for item in as_value if item)
            aggregate = transform.get('aggregate') or transform.get('joinaggregate') or []
            if isinstance(aggregate, list):
                for item in aggregate:
                    if isinstance(item, dict) and isinstance(item.get('as'), str):
                        fields.add(item['as'])
        return fields

    @classmethod
    def _validate_encoding(cls, spec: dict[str, Any], dataset_columns: set[str], errors: list[str]) -> dict[
        str, dict[str, Any]]:
        encoding = spec.get('encoding', {})
        if not isinstance(encoding, dict) or not encoding:
            errors.append('Specification encoding must be a non-empty object.')
            return {}
        derived_columns = cls._derived_fields_from_transforms(spec)
        allowed_fields = dataset_columns | derived_columns
        for channel, channel_spec in encoding.items():
            if channel not in _ALLOWED_CHANNELS:
                errors.append(f'Unsupported encoding channel: {channel}.')
                continue
            if isinstance(channel_spec, list):
                for item in channel_spec:
                    if isinstance(item, dict):
                        cls._validate_channel_spec(channel, item, allowed_fields, errors)
                    else:
                        errors.append(f'Encoding list item for channel {channel!r} must be an object.')
                continue
            if not isinstance(channel_spec, dict):
                errors.append(f'Encoding for channel {channel!r} must be an object.')
                continue
            cls._validate_channel_spec(channel, channel_spec, allowed_fields, errors)
        return encoding

    @staticmethod
    def _validate_channel_spec(channel: str, channel_spec: dict[str, Any], allowed_fields: set[str],
                               errors: list[str]) -> None:
        field = channel_spec.get('field')
        aggregate = channel_spec.get('aggregate')
        requires_field = channel not in {'detail', 'tooltip'} and not (
                channel == 'y' and aggregate == 'count') and not channel_spec.get('value')
        if requires_field:
            if not isinstance(field, str) or not field.strip():
                errors.append(f'Encoding.{channel}.field is required.')
            elif allowed_fields and field not in allowed_fields:
                errors.append(f'Encoding.{channel}.field references a missing dataset or derived column: {field}.')
        elif isinstance(field, str) and field.strip() and allowed_fields and field not in allowed_fields:
            errors.append(f'Encoding.{channel}.field references a missing dataset or derived column: {field}.')
        field_type = channel_spec.get('type')
        if field_type is not None and field_type not in _ALLOWED_TYPES:
            errors.append(f'Encoding.{channel}.type has unsupported value: {field_type}.')
        if aggregate == 'average':
            channel_spec['aggregate'] = 'mean'
            aggregate = 'mean'
        if aggregate is not None and aggregate not in _ALLOWED_AGGREGATES:
            errors.append(f'Encoding.{channel}.aggregate has unsupported value: {aggregate}.')

    @staticmethod
    def _validate_transforms(spec: dict[str, Any], dataset_columns: set[str], errors: list[str]) -> None:
        transforms = spec.get('transform', [])
        if transforms is None:
            return
        if not isinstance(transforms, list):
            errors.append('Specification transform must be a list when present.')
            return
        derived_fields: set[str] = set()
        for index, transform in enumerate(transforms):
            if not isinstance(transform, dict):
                errors.append(f'Transform at index {index} must be an object.')
                continue
            kind = transform.get('kind')
            if kind is None:
                for candidate in _ALLOWED_TRANSFORMS:
                    if candidate in transform:
                        kind = candidate
                        break
            if kind not in _ALLOWED_TRANSFORMS:
                errors.append(f'Unsupported transform kind at index {index}: {kind}.')
                continue
            for field in SpecValidatorService._transform_input_fields(transform):
                if field and dataset_columns and field not in dataset_columns and field not in derived_fields and field != 'count':
                    errors.append(f'Transform at index {index} references a missing field: {field}.')
            derived_fields.update(SpecValidatorService._transform_output_fields(transform))

    @staticmethod
    def _transform_input_fields(transform: dict[str, Any]) -> set[str]:
        fields: set[str] = set()
        for key in ('field', 'field_name'):
            value = transform.get(key)
            if isinstance(value, str):
                fields.add(value)
        for key in ('aggregate', 'joinaggregate', 'window'):
            value = transform.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and isinstance(item.get('field'), str):
                        fields.add(item['field'])
        return fields

    @staticmethod
    def _transform_output_fields(transform: dict[str, Any]) -> set[str]:
        fields: set[str] = set()
        value = transform.get('as')
        if isinstance(value, str):
            fields.add(value)
        elif isinstance(value, list):
            fields.update(str(item) for item in value if item)
        for key in ('aggregate', 'joinaggregate', 'window'):
            value = transform.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and isinstance(item.get('as'), str):
                        fields.add(item['as'])
        return fields

    @staticmethod
    def _validate_mark_specific_requirements(mark_type: str, encoding: dict[str, dict[str, Any]],
                                             errors: list[str]) -> None:
        x = encoding.get('x') if isinstance(encoding, dict) else None
        y = encoding.get('y') if isinstance(encoding, dict) else None
        if mark_type in _MARK_REQUIRING_XY:
            if not isinstance(x, dict) or (not x.get('field') and not x.get('value')):
                errors.append(f"Mark '{mark_type}' requires encoding.x.field or encoding.x.value.")
            if not isinstance(y, dict):
                errors.append(f"Mark '{mark_type}' requires encoding.y.")
            elif not y.get('field') and y.get('aggregate') != 'count' and not y.get('value'):
                errors.append(f"Mark '{mark_type}' requires encoding.y.field unless aggregate=count or value is set.")
        if mark_type == 'boxplot' and not isinstance(y, dict):
            errors.append('Boxplot requires a quantitative y encoding.')
        if mark_type == 'histogram':
            if not isinstance(x, dict) or not x.get('field'):
                errors.append('Histogram requires encoding.x.field for the binned measure.')
            if x and x.get('type') not in {None, 'quantitative'}:
                errors.append('Histogram encoding.x.type must be quantitative.')

    @staticmethod
    def _normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(spec)
        normalized.setdefault('$schema', 'https://vega.github.io/schema/vega-lite/v5.json')
        mark = normalized.get('mark')
        mark_type = mark.get('type') if isinstance(mark, dict) else mark
        if isinstance(mark_type, str) and mark_type.lower() == 'scatter':
            if isinstance(mark, dict):
                mark['type'] = 'point'
            else:
                normalized['mark'] = 'point'
        return normalized

    @classmethod
    def _validate_with_vega_runtime(cls, spec: dict[str, Any], dataset_path: Path) -> dict[str, Any]:
        try:
            import altair as alt
        except ImportError as exc:
            return {
                'is_valid_schema': False,
                'is_valid_scenegraph': False,
                'is_empty_scenegraph': True,
                'schema_error': f'Altair is required for strict schema validation: {exc}',
                'scenegraph_error': None,
            }
        try:
            import vl_convert as vlc  # type: ignore
        except ImportError as exc:
            return {
                'is_valid_schema': False,
                'is_valid_scenegraph': False,
                'is_empty_scenegraph': True,
                'schema_error': None,
                'scenegraph_error': f'vl-convert-python is required for strict scenegraph validation: {exc}',
            }
        try:
            df = read_dataframe(dataset_path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return {
                'is_valid_schema': False,
                'is_valid_scenegraph': False,
                'is_empty_scenegraph': True,
                'schema_error': f'Could not read dataset: {exc}',
                'scenegraph_error': f'Could not read dataset: {exc}',
            }

        schema_error = None
        scenegraph_error = None
        is_valid_schema = False
        is_valid_scenegraph = False
        is_empty_scenegraph = True

        try:
            alt.Chart.from_dict(cls._spec_add_data(spec, df.head()))
            is_valid_schema = True
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
            key = value.strip()
            if key and key not in seen:
                seen.add(key)
                result.append(key)
        return result
