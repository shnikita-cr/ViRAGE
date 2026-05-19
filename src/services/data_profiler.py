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
        column_errors: list[dict[str, Any]] = []

        row_count = int(len(df))
        col_count = int(len(df.columns))
        sample_seed = int(getattr(runtime.settings, "data_profile_sample_seed", 42))
        sample_size = max(1, int(getattr(runtime.settings, "data_profile_sample_size", 10)))
        column_name_map = self._build_unique_column_name_map([str(column) for column in df.columns])

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
            safe_name = column_name_map[column_name]
            if safe_name != column_name:
                cleaning_hints.append(f"Column '{column_name}' can be normalized to '{safe_name}' for renderer safety.")

            try:
                column_profile, column_quality_notes = self._profile_column(
                    column_name=column_name,
                    safe_name=safe_name,
                    series=df[column],
                    row_count=row_count,
                )
                quality_notes.extend(column_quality_notes)
            except Exception as exc:  # noqa: BLE001 - column-level degradation is intentional.
                error_payload = self._column_error_payload(column_name=column_name, series=df[column], exc=exc)
                column_errors.append(error_payload)
                quality_notes.append(
                    f"Column '{column_name}' could not be fully profiled and was treated as categorical: {type(exc).__name__}: {exc}"
                )
                column_profile = self._degraded_column_profile(
                    column_name=column_name,
                    safe_name=safe_name,
                    series=df[column],
                    row_count=row_count,
                )

            columns.append(column_profile)
            semantic_dtype = column_profile.dtype

            if semantic_dtype == "numeric":
                numeric.append(column_name)
                field_roles[column_name] = "measure"
            elif semantic_dtype == "datetime":
                time_like.append(column_name)
                field_roles[column_name] = "temporal"
            else:
                categorical.append(column_name)
                field_roles[column_name] = "dimension"

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
        if column_errors:
            complexity_hints.append("degraded_profile")
            cleaning_hints.append("review_profile_column_errors")
        schema_hints.extend([f"{column.name}:{column.dtype}" for column in columns])

        profile = DataProfile(
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
            profile_status="degraded" if column_errors else "ok",
            column_errors=column_errors,
            sample_strategy="random",
            sample_seed=sample_seed,
            sample_size=min(sample_size, row_count) if row_count else 0,
        )

        return profile

    def _profile_column(
            self,
            *,
            column_name: str,
            safe_name: str,
            series: pd.Series,
            row_count: int,
            sample_seed: int = 42,
            sample_size: int = 10,
    ) -> tuple[DataColumnProfile, list[str]]:
        quality_notes: list[str] = []
        missing_ratio = float(series.isna().mean()) if row_count else 0.0
        unique_count = int(series.nunique(dropna=True))
        semantic_dtype = self._semantic_dtype(column_name, series)
        min_value, max_value, min_max_note = self._min_max(column_name, series, semantic_dtype)
        if min_max_note:
            quality_notes.append(min_max_note)
        samples = self._sample_values(series, sample_seed=sample_seed, sample_size=sample_size)
        outlier_count, outlier_ratio = self._outlier_stats(series, semantic_dtype)
        is_identifier = self._looks_identifier(column_name, unique_count, row_count)
        is_high_cardinality = semantic_dtype == "categorical" and unique_count > max(50, int(row_count * 0.5))

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

        return (
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
            ),
            quality_notes,
        )

    def _semantic_dtype(self, column: str, series: pd.Series) -> str:
        non_null = series.dropna()
        if pd.api.types.is_bool_dtype(series) or self._looks_boolean_like(non_null):
            return "boolean"
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
        if series.empty or pd.api.types.is_bool_dtype(series):
            return False
        converted = pd.to_numeric(series, errors="coerce")
        return float(converted.notna().mean()) >= 0.9

    @staticmethod
    def _looks_boolean_like(series: pd.Series) -> bool:
        if series.empty:
            return False
        normalized = series.astype(str).str.strip().str.lower()
        allowed = {"true", "false", "yes", "no", "y", "n", "t", "f"}
        return bool(normalized.isin(allowed).all())

    def _can_parse_datetime_like(self, column: str, series: pd.Series) -> bool:
        if series.empty or pd.api.types.is_bool_dtype(series) or self._looks_boolean_like(series):
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
        if pd.api.types.is_bool_dtype(series):
            return False
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return False
        if (numeric.between(0, 99).mean() >= 0.9) or (numeric.between(1000, 9999).mean() >= 0.9):
            return True
        return False

    @classmethod
    def _min_max(cls, column: str, series: pd.Series, semantic_dtype: str) -> tuple[Any | None, Any | None, str | None]:
        non_null = series.dropna()
        if non_null.empty:
            return None, None, None
        try:
            if semantic_dtype == "numeric":
                numeric = pd.to_numeric(non_null, errors="coerce").dropna().astype("float64")
                if numeric.empty:
                    return None, None, None
                return numeric.min().item(), numeric.max().item(), None
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
                    return None, None, None
                return converted.min().isoformat(), converted.max().isoformat(), None
            return None, None, None
        except (TypeError, ValueError, OverflowError) as exc:
            return None, None, f"Column {column!r} min/max could not be computed: {exc}"

    @staticmethod
    def _sample_values(series: pd.Series, *, sample_seed: int = 42, sample_size: int = 10) -> list[Any]:
        non_null = series.dropna()
        if non_null.empty:
            return []
        size = min(max(1, int(sample_size)), len(non_null))
        try:
            sampled = non_null.sample(n=size, random_state=int(sample_seed))
        except Exception:
            sampled = non_null.head(size)
        values = []
        for value in sampled.tolist():
            if hasattr(value, "item"):
                value = value.item()
            values.append(value)
        return values

    @staticmethod
    def _outlier_stats(series: pd.Series, semantic_dtype: str) -> tuple[int, float]:
        if semantic_dtype != "numeric" or pd.api.types.is_bool_dtype(series):
            return 0, 0.0
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return 0, 0.0
        numeric = numeric.astype("float64")
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

    @classmethod
    def _build_unique_column_name_map(cls, columns: list[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        used: set[str] = set()

        for original in columns:
            base = cls._safe_column_name(original)
            candidate = base
            suffix = 2

            while candidate in used:
                candidate = f"{base}_{suffix}"
                suffix += 1

            used.add(candidate)
            mapping[original] = candidate

        return mapping

    @staticmethod
    def _safe_column_name(name: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip())
        safe = re.sub(r"_+", "_", safe).strip("_")
        if safe and safe[0].isdigit():
            safe = f"col_{safe}"
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

    def _degraded_column_profile(
            self,
            *,
            column_name: str,
            safe_name: str,
            series: pd.Series,
            row_count: int,
            sample_seed: int = 42,
            sample_size: int = 10,
    ) -> DataColumnProfile:
        return DataColumnProfile(
            name=column_name,
            original_name=column_name,
            safe_name=safe_name,
            dtype="categorical",
            missing_ratio=float(series.isna().mean()) if row_count else 0.0,
            unique_count=int(series.nunique(dropna=True)),
            sample_values=self._sample_values(series, sample_seed=sample_seed, sample_size=sample_size),
            outlier_count=0,
            outlier_ratio=0.0,
            is_identifier=self._looks_identifier(column_name, int(series.nunique(dropna=True)), row_count),
            is_high_cardinality=False,
        )

    @staticmethod
    def _column_error_payload(column_name: str, series: pd.Series, exc: Exception) -> dict[str, Any]:
        return {
            "failure_class": "data_profile_error",
            "subreason": "column_profile_failed",
            "recoverable": True,
            "column": column_name,
            "raw_dtype": str(series.dtype),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    @staticmethod
    def _save_column_error(runtime: RuntimeContext, error_payload: dict[str, Any]) -> None:
        return

    @staticmethod
    def _save_profile_error_summary(runtime: RuntimeContext, profile: DataProfile) -> None:
        return

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
