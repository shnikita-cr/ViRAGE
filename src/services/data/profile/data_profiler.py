from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd

from src.domain.models import DataColumnProfile, DataProfile
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService

_TEMPORAL_NAME_RE = re.compile(r"(^|[_\s-])(date|time|timestamp|year|month|day)([_\s-]|$)", re.IGNORECASE)
_ID_NAME_RE = re.compile(r"(^|[_\s-])(id|uuid|guid|key)([_\s-]|$)", re.IGNORECASE)
_MISSING_LIKE_VALUES = {"", "-", "--", "---", "na", "n/a", "nan", "none", "null", "missing"}


class DataProfilerService(BaseService):
    def invoke(self, data_path: str, runtime: RuntimeContext) -> DataProfile:
        df = self._ensure_unique_columns(runtime.read_dataframe(data_path))
        row_count = int(len(df))
        col_count = int(len(df.columns))
        sample_seed, sample_size = self._profile_settings(runtime)
        column_name_map = self._build_unique_column_name_map([str(column) for column in df.columns])

        quality_notes: list[str] = []
        complexity_hints: list[str] = []
        errors: list[dict[str, Any]] = []
        self._append_dataset_notes(
            df=df,
            row_count=row_count,
            col_count=col_count,
            quality_notes=quality_notes,
        )

        columns = self._profile_columns(
            df=df,
            column_name_map=column_name_map,
            row_count=row_count,
            sample_seed=sample_seed,
            sample_size=sample_size,
            quality_notes=quality_notes,
            errors=errors,
        )
        self._append_profile_summary(
            columns=columns,
            row_count=row_count,
            col_count=col_count,
            errors=errors,
            quality_notes=quality_notes,
            complexity_hints=complexity_hints,
        )

        return DataProfile(
            row_count=row_count,
            col_count=col_count,
            columns=columns,
            quality_notes=self._dedupe(quality_notes),
            complexity_hints=self._dedupe(complexity_hints),
            data_complexity="large" if row_count > 100_000 or col_count > 30 else "standard",
            profile_status="degraded" if errors else "ok",
            errors=errors,
            sample_strategy="random",
            sample_seed=sample_seed,
            sample_size=min(sample_size, row_count) if row_count else 0,
            source_format=df.attrs.get("source_format"),
            source_encoding=df.attrs.get("source_encoding"),
        )

    @staticmethod
    def _profile_settings(runtime: RuntimeContext) -> tuple[int, int]:
        sample_seed = int(getattr(runtime.settings, "data_profile_sample_seed", 42))
        sample_size = max(1, int(getattr(runtime.settings, "data_profile_sample_size", 5)))
        return sample_seed, sample_size

    @staticmethod
    def _append_dataset_notes(
            *,
            df: pd.DataFrame,
            row_count: int,
            col_count: int,
            quality_notes: list[str],
    ) -> None:
        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows:
            quality_notes.append(
                f"Dataset contains {duplicate_rows} duplicated rows. They are preserved by default because rows may be valid records."
            )
        if row_count == 0:
            quality_notes.append("Dataset is empty.")
        if col_count == 0:
            quality_notes.append("Dataset has no columns.")

    def _profile_columns(
            self,
            *,
            df: pd.DataFrame,
            column_name_map: dict[str, str],
            row_count: int,
            sample_seed: int,
            sample_size: int,
            quality_notes: list[str],
            errors: list[dict[str, Any]],
    ) -> list[DataColumnProfile]:
        columns: list[DataColumnProfile] = []
        for column in df.columns:
            column_name = str(column)
            column_profile = self._profile_column_with_degradation(
                column_name=column_name,
                safe_name=column_name_map[column_name],
                series=df[column],
                row_count=row_count,
                sample_seed=sample_seed,
                sample_size=sample_size,
                quality_notes=quality_notes,
                errors=errors,
            )
            columns.append(column_profile)
        return columns

    def _profile_column_with_degradation(
            self,
            *,
            column_name: str,
            safe_name: str,
            series: pd.Series,
            row_count: int,
            sample_seed: int,
            sample_size: int,
            quality_notes: list[str],
            errors: list[dict[str, Any]],
    ) -> DataColumnProfile:
        try:
            column_profile, column_quality_notes = self._profile_column(
                column_name=column_name,
                safe_name=safe_name,
                series=series,
                row_count=row_count,
                sample_seed=sample_seed,
                sample_size=sample_size,
            )
            quality_notes.extend(column_quality_notes)
            return column_profile
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as exc:  # noqa: BLE001 - column-level degradation is intentional.
            errors.append(self._column_error_payload(column_name=column_name, series=series, exc=exc))
            quality_notes.append(
                f"Column '{column_name}' could not be fully profiled and was treated as categorical: {type(exc).__name__}: {exc}"
            )
            return self._degraded_column_profile(
                column_name=column_name,
                safe_name=safe_name,
                series=series,
                row_count=row_count,
                sample_seed=sample_seed,
                sample_size=sample_size,
            )

    @staticmethod
    def _append_profile_summary(
            *,
            columns: list[DataColumnProfile],
            row_count: int,
            col_count: int,
            errors: list[dict[str, Any]],
            quality_notes: list[str],
            complexity_hints: list[str],
    ) -> None:
        if not any(column.role == "measure" for column in columns):
            quality_notes.append("No numeric measure columns detected; numeric chart options may be limited.")
        temporal_columns = [column.name for column in columns if column.role == "temporal"]
        if temporal_columns:
            quality_notes.append(f"Detected time-like columns: {', '.join(temporal_columns)}.")
        if row_count > 100_000:
            quality_notes.append("Large dataset detected; sampling or aggregation may be required downstream.")
            complexity_hints.append("large_dataset")
        if col_count > 30:
            complexity_hints.append("wide_dataset")
        if errors:
            complexity_hints.append("degraded_profile")

    def _profile_column(
            self,
            *,
            column_name: str,
            safe_name: str,
            series: pd.Series,
            row_count: int,
            sample_seed: int = 42,
            sample_size: int = 5,
    ) -> tuple[DataColumnProfile, list[str]]:
        quality_notes: list[str] = []
        raw_dtype = str(series.dtype)
        missing_ratio = float(series.isna().mean()) if row_count else 0.0
        missing_like_ratio = self._missing_like_ratio(series)
        unique_count = int(series.nunique(dropna=True))
        semantic_dtype = self._semantic_dtype(column_name, series)
        min_value, max_value, min_max_note = self._min_max(column_name, series, semantic_dtype)
        if min_max_note:
            quality_notes.append(min_max_note)
        samples = self._sample_values(series, sample_seed=sample_seed, sample_size=sample_size)
        outlier_count, outlier_ratio = self._outlier_stats(series, semantic_dtype)
        is_identifier = self._looks_identifier(column_name, unique_count, row_count)
        is_high_cardinality = semantic_dtype == "categorical" and unique_count > max(50, int(row_count * 0.5))
        flags = self._field_quality_flags(
            column_name=column_name,
            series=series,
            semantic_dtype=semantic_dtype,
            missing_ratio=missing_ratio,
            missing_like_ratio=missing_like_ratio,
            unique_count=unique_count,
            row_count=row_count,
            is_identifier=is_identifier,
            is_high_cardinality=is_high_cardinality,
        )
        recommended_preparation = self._recommended_preparation(flags, semantic_dtype)

        if missing_ratio >= 0.5:
            quality_notes.append(f"Column '{column_name}' has high missing ratio ({missing_ratio:.0%}).")
        if missing_like_ratio >= 0.02:
            quality_notes.append(
                f"Column '{column_name}' contains missing-like string values ({missing_like_ratio:.0%}); treat them as nulls before numeric/statistical use."
            )
        if unique_count <= 1 and row_count > 0:
            quality_notes.append(f"Column '{column_name}' is constant or nearly constant.")
        if is_identifier:
            quality_notes.append(f"Column '{column_name}' looks like an identifier.")
        if is_high_cardinality:
            quality_notes.append(f"Column '{column_name}' has high cardinality for a categorical field.")
        if "numeric_string" in flags:
            quality_notes.append(f"Column '{column_name}' looks numeric but is stored as text.")
        if outlier_count:
            quality_notes.append(
                f"Column '{column_name}' has {outlier_count} potential numeric outliers ({outlier_ratio:.1%})."
            )

        return (
            DataColumnProfile(
                name=column_name,
                safe_name=safe_name,
                dtype=semantic_dtype,
                role=self._field_role(semantic_dtype, is_identifier),
                missing_ratio=missing_ratio,
                unique_count=unique_count,
                min_value=min_value,
                max_value=max_value,
                sample_values=samples,
                outlier_count=outlier_count,
                outlier_ratio=outlier_ratio,
                is_identifier=is_identifier,
                is_high_cardinality=is_high_cardinality,
                raw_dtype=raw_dtype,
                missing_like_ratio=missing_like_ratio,
                quality_flags=flags,
                preparation_hints=recommended_preparation,
            ),
            quality_notes,
        )

    @staticmethod
    def _field_role(semantic_dtype: str, is_identifier: bool) -> str:
        if is_identifier:
            return "identifier"
        if semantic_dtype == "numeric":
            return "measure"
        if semantic_dtype == "datetime":
            return "temporal"
        if semantic_dtype == "categorical" or semantic_dtype == "boolean":
            return "dimension"
        return "unknown"

    @staticmethod
    def _missing_like_ratio(series: pd.Series) -> float:
        non_null = series.dropna()
        if non_null.empty:
            return 0.0
        normalized = non_null.astype(str).str.strip().str.lower()
        return float(normalized.isin(_MISSING_LIKE_VALUES).mean())

    @classmethod
    def _field_quality_flags(
            cls,
            *,
            column_name: str,
            series: pd.Series,
            semantic_dtype: str,
            missing_ratio: float,
            missing_like_ratio: float,
            unique_count: int,
            row_count: int,
            is_identifier: bool,
            is_high_cardinality: bool,
    ) -> list[str]:
        flags: list[str] = []
        if missing_ratio >= 0.5:
            flags.append("many_missing")
        if missing_like_ratio >= 0.02:
            flags.append("missing_like_strings")
        if is_identifier:
            flags.append("identifier_like")
        if is_high_cardinality:
            flags.extend(["high_cardinality", "unsafe_for_color"])
        if cls._is_temporal_name(column_name) or semantic_dtype == "datetime":
            flags.append("year_like" if cls._is_year_like(column_name, series.dropna()) else "temporal_like")
        if semantic_dtype == "numeric" and not pd.api.types.is_numeric_dtype(series):
            flags.append("numeric_string")
        if semantic_dtype == "numeric" and not is_identifier:
            flags.append("good_for_measure")
        if semantic_dtype == "datetime":
            flags.append("good_for_temporal_axis")
        if semantic_dtype in {"categorical", "boolean"} and not is_identifier and not is_high_cardinality:
            flags.append("good_for_grouping")
            flags.append("safe_for_color")
        if is_high_cardinality:
            flags.append("unsafe_for_color")
        if unique_count <= 1 and row_count > 0:
            flags.append("constant_or_nearly_constant")
        return cls._dedupe(flags)

    @staticmethod
    def _recommended_preparation(flags: list[str], semantic_dtype: str) -> list[str]:
        recommendations: list[str] = []
        if "missing_like_strings" in flags:
            recommendations.append("normalize_missing_like_strings")
        if "numeric_string" in flags:
            recommendations.append("coerce_text_to_numeric")
        if semantic_dtype == "datetime" or "year_like" in flags:
            recommendations.append("normalize_temporal_values")
        if "unsafe_for_color" in flags:
            recommendations.append("avoid_color_encoding_without_filtering_or_faceting")
        return recommendations

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
    def _sample_values(series: pd.Series, *, sample_seed: int = 42, sample_size: int = 5) -> list[Any]:
        non_null = series.dropna()
        if non_null.empty:
            return []
        size = min(max(1, int(sample_size)), len(non_null))
        try:
            sampled = non_null.sample(n=size, random_state=int(sample_seed))
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError):
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
            sample_size: int = 5,
    ) -> DataColumnProfile:
        return DataColumnProfile(
            name=column_name,
            safe_name=safe_name,
            dtype="categorical",
            role="dimension",
            missing_ratio=float(series.isna().mean()) if row_count else 0.0,
            unique_count=int(series.nunique(dropna=True)),
            sample_values=self._sample_values(series, sample_seed=sample_seed, sample_size=sample_size),
            outlier_count=0,
            outlier_ratio=0.0,
            is_identifier=self._looks_identifier(column_name, int(series.nunique(dropna=True)), row_count),
            is_high_cardinality=False,
            raw_dtype=str(series.dtype),
            missing_like_ratio=self._missing_like_ratio(series),
            quality_flags=["degraded_profile"],
            preparation_hints=["review_profile_column_errors"],
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
