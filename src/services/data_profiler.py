from __future__ import annotations

from pathlib import Path

import pandas as pd
import warnings

from src.domain.models import DataColumnProfile, DataProfile
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class DataProfilerService(BaseService):
    def invoke(self, data_path: str, runtime: RuntimeContext) -> DataProfile:
        path = Path(data_path)
        df = self._read_frame(path)

        columns: list[DataColumnProfile] = []
        numeric: list[str] = []
        categorical: list[str] = []
        time_like: list[str] = []
        quality_notes: list[str] = []

        row_count = int(len(df))
        col_count = int(len(df.columns))

        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows:
            quality_notes.append(f"Dataset contains {duplicate_rows} duplicated rows.")
        if row_count == 0:
            quality_notes.append("Dataset is empty.")
        if col_count == 0:
            quality_notes.append("Dataset has no columns.")

        for column in df.columns:
            series = df[column]
            missing_ratio = float(series.isna().mean()) if row_count else 0.0
            unique_count = int(series.nunique(dropna=True))
            semantic_dtype = self._semantic_dtype(column, series)
            columns.append(
                DataColumnProfile(
                    name=str(column),
                    dtype=semantic_dtype,
                    missing_ratio=missing_ratio,
                    unique_count=unique_count,
                )
            )

            if semantic_dtype == "numeric":
                numeric.append(str(column))
            elif semantic_dtype == "datetime":
                time_like.append(str(column))
            else:
                categorical.append(str(column))

            if missing_ratio >= 0.5:
                quality_notes.append(f"Column '{column}' has high missing ratio ({missing_ratio:.0%}).")
            if unique_count <= 1 and row_count > 0:
                quality_notes.append(f"Column '{column}' is constant or nearly constant.")
            if unique_count == row_count and row_count > 10 and semantic_dtype != "datetime":
                lowered = str(column).lower()
                if any(token in lowered for token in ["id", "uuid", "guid", "key"]):
                    quality_notes.append(f"Column '{column}' looks like an identifier.")
            if semantic_dtype == "categorical" and unique_count > max(50, int(row_count * 0.5)):
                quality_notes.append(f"Column '{column}' has high cardinality for a categorical field.")

        if not numeric:
            quality_notes.append("No numeric columns detected; numeric chart options may be limited.")
        if time_like:
            quality_notes.append(f"Detected time-like columns: {', '.join(time_like)}.")
        if row_count > 100_000:
            quality_notes.append("Large dataset detected; sampling or aggregation may be required downstream.")

        return DataProfile(
            row_count=row_count,
            col_count=col_count,
            columns=columns,
            likely_numeric_columns=numeric,
            likely_categorical_columns=categorical,
            likely_time_columns=time_like,
            quality_notes=self._dedupe(quality_notes),
        )

    @staticmethod
    def _read_frame(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
        raise ValueError(f"Unsupported data format: {suffix}")

    def _semantic_dtype(self, column: str, series: pd.Series) -> str:
        lowered = str(column).lower()
        non_null = series.dropna()
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if any(token in lowered for token in ["date", "time", "timestamp", "year", "month", "day"]):
            if self._can_parse_datetime(non_null):
                return "datetime"
        if self._looks_numeric(non_null):
            return "numeric"
        if self._can_parse_datetime(non_null):
            return "datetime"
        return "categorical"

    @staticmethod
    def _looks_numeric(series: pd.Series) -> bool:
        if series.empty:
            return False
        converted = pd.to_numeric(series, errors="coerce")
        return float(converted.notna().mean()) >= 0.9

    @staticmethod
    def _can_parse_datetime(series: pd.Series) -> bool:
        if series.empty:
            return False
        sample = series.astype(str).head(50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted = pd.to_datetime(sample, errors="coerce", utc=False)
        return float(converted.notna().mean()) >= 0.8

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
