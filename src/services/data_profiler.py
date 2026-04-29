from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd

from src.domain.models import DataColumnProfile, DataProfile
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.data import read_dataframe

_TEMPORAL_NAME_RE = re.compile(r"(^|[_\s-])(date|time|timestamp|year|month|day)([_\s-]|$)", re.IGNORECASE)
_ID_NAME_RE = re.compile(r"(^|[_\s-])(id|uuid|guid|key)([_\s-]|$)", re.IGNORECASE)


class DataProfilerService(BaseService):
    def invoke(self, data_path: str, runtime: RuntimeContext) -> DataProfile:
        df = read_dataframe(data_path)
        df = self._ensure_unique_columns(df)

        columns: list[DataColumnProfile] = []
        numeric: list[str] = []
        categorical: list[str] = []
        time_like: list[str] = []
        quality_notes: list[str] = []
        field_roles: dict[str, str] = {}
        schema_hints: list[str] = []
        complexity_hints: list[str] = []
        cleaning_hints: list[str] = []
        column_name_map: dict[str, str] = {}

        row_count = int(len(df))
        col_count = int(len(df.columns))

        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows:
            quality_notes.append(
                f"Dataset contains {duplicate_rows} duplicated rows. They are preserved by default because rows may be valid records."
            )
        if row_count == 0:
            quality_notes.append("Dataset is empty.")
        if col_count == 0:
            quality_notes.append("Dataset has no columns.")

        for column in df.columns:
            column_name = str(column)
            safe_name = self._safe_column_name(column_name)
            column_name_map[column_name] = safe_name
            if safe_name != column_name:
                cleaning_hints.append(f"Column '{column_name}' can be normalized to '{safe_name}' for renderer safety.")

            series = df[column]
            missing_ratio = float(series.isna().mean()) if row_count else 0.0
            unique_count = int(series.nunique(dropna=True))
            semantic_dtype = self._semantic_dtype(column_name, series)
            min_value, max_value = self._min_max(column_name, series, semantic_dtype)
            samples = self._sample_values(series)
            outlier_count, outlier_ratio = self._outlier_stats(series, semantic_dtype)
            is_identifier = self._looks_identifier(column_name, unique_count, row_count)
            is_high_cardinality = semantic_dtype == "categorical" and unique_count > max(50, int(row_count * 0.5))

            columns.append(
                DataColumnProfile(
                    name=column_name,
                    original_name=column_name,
                    safe_name=safe_name,
                    dtype=semantic_dtype,
                    missing_ratio=missing_ratio,
                    unique_count=unique_count,
                    min_value=min_value,
                    max_value=max_value,
                    sample_values=samples,
                    outlier_count=outlier_count,
                    outlier_ratio=outlier_ratio,
                    is_identifier=is_identifier,
                    is_high_cardinality=is_high_cardinality,
                )
            )

            if semantic_dtype == "numeric":
                numeric.append(column_name)
                field_roles[column_name] = "measure"
            elif semantic_dtype == "datetime":
                time_like.append(column_name)
                field_roles[column_name] = "temporal"
            else:
                categorical.append(column_name)
                field_roles[column_name] = "dimension"

            if missing_ratio >= 0.5:
                quality_notes.append(f"Column '{column_name}' has high missing ratio ({missing_ratio:.0%}).")
            if unique_count <= 1 and row_count > 0:
                quality_notes.append(f"Column '{column_name}' is constant or nearly constant.")
            if is_identifier:
                quality_notes.append(f"Column '{column_name}' looks like an identifier.")
            if is_high_cardinality:
                quality_notes.append(f"Column '{column_name}' has high cardinality for a categorical field.")
            if outlier_count:
                quality_notes.append(
                    f"Column '{column_name}' has {outlier_count} potential numeric outliers ({outlier_ratio:.1%})."
                )

        if not numeric:
            quality_notes.append("No numeric columns detected; numeric chart options may be limited.")
        if time_like:
            quality_notes.append(f"Detected time-like columns: {', '.join(time_like)}.")
        if row_count > 100_000:
            quality_notes.append("Large dataset detected; sampling or aggregation may be required downstream.")
            complexity_hints.append("large_dataset")
        if col_count > 30:
            complexity_hints.append("wide_dataset")
        if duplicate_rows:
            cleaning_hints.append("preserve_row_multiplicity")
        schema_hints.extend([f"{column.name}:{column.dtype}" for column in columns])

        return DataProfile(
            row_count=row_count,
            col_count=col_count,
            columns=columns,
            likely_numeric_columns=numeric,
            likely_categorical_columns=categorical,
            likely_time_columns=time_like,
            quality_notes=self._dedupe(quality_notes),
            typed_columns=[f"{column.name}:{column.dtype}" for column in columns],
            field_roles=field_roles,
            schema_hints=self._dedupe(schema_hints),
            complexity_hints=self._dedupe(complexity_hints),
            cleaning_hints=self._dedupe(cleaning_hints),
            column_name_map=column_name_map,
            data_complexity="large" if row_count > 100_000 or col_count > 30 else "standard",
        )

    def _semantic_dtype(self, column: str, series: pd.Series) -> str:
        non_null = series.dropna()
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        # Semantic temporal names are checked before numeric dtype so columns such as Year=1970 or 70 are treated as time.
        if self._is_temporal_name(column) and self._can_parse_datetime_like(column, non_null):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if self._looks_numeric(non_null):
            return "numeric"
        if self._can_parse_datetime_like(column, non_null):
            return "datetime"
        return "categorical"

    @staticmethod
    def _is_temporal_name(column: str) -> bool:
        lowered = str(column).lower().strip()
        return bool(_TEMPORAL_NAME_RE.search(lowered) or lowered in {"year", "month", "date", "time"})

    @staticmethod
    def _looks_numeric(series: pd.Series) -> bool:
        if series.empty:
            return False
        converted = pd.to_numeric(series, errors="coerce")
        return float(converted.notna().mean()) >= 0.9

    def _can_parse_datetime_like(self, column: str, series: pd.Series) -> bool:
        if series.empty:
            return False
        if self._is_year_like(column, series):
            return True
        sample = series.astype(str).head(50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted = pd.to_datetime(sample, errors="coerce", utc=False)
        return float(converted.notna().mean()) >= 0.8

    @staticmethod
    def _is_year_like(column: str, series: pd.Series) -> bool:
        if "year" not in str(column).lower():
            return False
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return False
        if (numeric.between(0, 99).mean() >= 0.9) or (numeric.between(1000, 9999).mean() >= 0.9):
            return True
        return False

    @classmethod
    def _min_max(cls, column: str, series: pd.Series, semantic_dtype: str) -> tuple[Any | None, Any | None]:
        non_null = series.dropna()
        if non_null.empty:
            return None, None
        try:
            if semantic_dtype == "numeric":
                numeric = pd.to_numeric(non_null, errors="coerce").dropna()
                if numeric.empty:
                    return None, None
                return numeric.min().item(), numeric.max().item()
            if semantic_dtype == "datetime":
                if cls._is_year_like(column, non_null):
                    numeric = pd.to_numeric(non_null, errors="coerce").dropna()

                    def expand_year(value: float) -> int:
                        rounded = int(round(value))
                        if 0 <= rounded <= 29:
                            return 2000 + rounded
                        if 30 <= rounded <= 99:
                            return 1900 + rounded
                        return rounded

                    years = numeric.map(expand_year)
                    converted = pd.to_datetime(years.astype("Int64").astype("string") + "-01-01",
                                               errors="coerce").dropna()
                else:
                    converted = pd.to_datetime(non_null.head(1000), errors="coerce").dropna()
                if converted.empty:
                    return None, None
                return converted.min().isoformat(), converted.max().isoformat()
            return None, None
        except Exception:
            return None, None

    @staticmethod
    def _sample_values(series: pd.Series) -> list[Any]:
        values = []
        for value in series.dropna().head(5).tolist():
            if hasattr(value, "item"):
                value = value.item()
            values.append(value)
        return values

    @staticmethod
    def _outlier_stats(series: pd.Series, semantic_dtype: str) -> tuple[int, float]:
        if semantic_dtype != "numeric":
            return 0, 0.0
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) < 8:
            return 0, 0.0
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0, 0.0
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((numeric < lower) | (numeric > upper)).sum())
        return count, float(count / len(numeric)) if len(numeric) else 0.0

    @staticmethod
    def _looks_identifier(column: str, unique_count: int, row_count: int) -> bool:
        if row_count <= 10:
            return False
        return bool(_ID_NAME_RE.search(str(column).lower())) and unique_count >= int(row_count * 0.8)

    @staticmethod
    def _safe_column_name(name: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip())
        safe = re.sub(r"_+", "_", safe).strip("_")
        return safe or "column"

    @staticmethod
    def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
        if not df.columns.duplicated().any():
            return df
        counts: dict[str, int] = {}
        new_columns: list[str] = []
        for name in df.columns:
            key = str(name)
            counts[key] = counts.get(key, 0) + 1
            new_columns.append(key if counts[key] == 1 else f"{key}__dup{counts[key] - 1}")
        copy = df.copy()
        copy.columns = new_columns
        return copy

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(value.strip())
        return result
