from __future__ import annotations

import pandas as pd

from src.domain.models import DataPreparationResult, DataProfile, QueryRequestAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.data.preparation.metric_severity_transformer import MetricSeverityTransformer


class DataPreparationService(BaseService):
    def invoke(
            self,
            data_path: str,
            data_profile: DataProfile,
            request_analysis: QueryRequestAnalysisResult,
            run_id: str,
            runtime: RuntimeContext,
    ) -> DataPreparationResult:
        df = runtime.read_dataframe(data_path)
        df = _ensure_unique_columns(df)
        operations = ["preserve_row_multiplicity"]

        original_columns = [str(column) for column in df.columns]
        column_name_map = _build_column_name_map(data_profile=data_profile, original_columns=original_columns)
        reverse_column_name_map = {safe: original for original, safe in column_name_map.items()}
        renamed_column_count = sum(1 for original, safe in column_name_map.items() if original != safe)

        if renamed_column_count:
            df = df.rename(columns=column_name_map)
            operations.append(f"safe_column_mapping:{renamed_column_count}")

        for column_profile in data_profile.temporal_columns():
            original_column = column_profile.name
            safe_column = column_name_map.get(original_column, original_column)
            if safe_column in df.columns:
                df[safe_column] = _parse_temporal(original_column, df[safe_column])
                operations.append(f"to_datetime:{original_column}->{safe_column}")

        metric_semantics = _map_metric_semantics(
            getattr(request_analysis, "metric_semantics", {}) or {},
            column_name_map=column_name_map,
        )
        selected_fields = [column_name_map.get(field, field) for field in list(getattr(request_analysis, "selected_fields", []) or [])]
        severity_result = MetricSeverityTransformer().apply(
            df,
            metric_semantics=metric_semantics,
            ranking_strategy=getattr(request_analysis, "ranking_strategy", None),
            selected_fields=selected_fields,
            max_ranked_rows=runtime.settings.problematic_item_top_n,
        )
        df = severity_result.frame
        operations.extend(severity_result.operations)

        safe_columns = [str(column) for column in df.columns]

        output_path = runtime.next_artifact_path("cleaned_data.csv", run_id=run_id)
        df.to_csv(output_path, index=False)
        return DataPreparationResult(
            output_path=output_path.as_posix(),
            operations=operations,
            row_count=len(df),
            col_count=len(df.columns),
            column_name_map=column_name_map,
            reverse_column_name_map=reverse_column_name_map,
            original_columns=original_columns,
            safe_columns=safe_columns,
            renamed_column_count=renamed_column_count,
        )



def _map_metric_semantics(value: dict, *, column_name_map: dict[str, str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, direction in dict(value or {}).items():
        original = str(key).strip()
        if not original:
            continue
        result[column_name_map.get(original, original)] = direction
    return result

def _build_column_name_map(*, data_profile: DataProfile, original_columns: list[str]) -> dict[str, str]:
    mapping = {str(original): str(safe) for original, safe in data_profile.original_to_safe_map().items()}
    missing = [column for column in original_columns if column not in mapping]
    if missing:
        generated = _build_unique_safe_mapping(missing, used=set(mapping.values()))
        mapping.update(generated)
    return {column: mapping.get(column, column) for column in original_columns}


def _build_unique_safe_mapping(columns: list[str], *, used: set[str] | None = None) -> dict[str, str]:
    used = set(used or set())
    mapping: dict[str, str] = {}

    for original in columns:
        base = _safe_column_name(original)
        candidate = base
        suffix = 2

        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1

        used.add(candidate)
        mapping[original] = candidate

    return mapping


def _safe_column_name(name: str) -> str:
    import re

    safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if safe and safe[0].isdigit():
        safe = f"col_{safe}"
    return safe or "column"


def _parse_temporal(column: str, series: pd.Series) -> pd.Series:
    if "year" in column.lower():
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            years = numeric.round().astype("Int64").map(_expand_year)
            return pd.to_datetime(years.astype("string") + "-01-01", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _expand_year(value):
    if pd.isna(value):
        return pd.NA
    year = int(value)
    if 0 <= year <= 29:
        return 2000 + year
    if 30 <= year <= 99:
        return 1900 + year
    return year


def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        copy = df.copy()
        copy.columns = [str(column) for column in copy.columns]
        return copy

    counts: dict[str, int] = {}
    new_columns: list[str] = []
    for name in df.columns:
        key = str(name)
        counts[key] = counts.get(key, 0) + 1
        new_columns.append(key if counts[key] == 1 else f"{key}__dup{counts[key] - 1}")
    copy = df.copy()
    copy.columns = new_columns
    return copy


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
