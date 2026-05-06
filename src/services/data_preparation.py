from __future__ import annotations

import pandas as pd

from src.domain.models import DataPreparationResult, DataProfile, QueryUnderstandingResult, RequestAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.data import read_dataframe


class DataPreparationService(BaseService):
    def invoke(
            self,
            data_path: str,
            data_profile: DataProfile,
            request_analysis: RequestAnalysisResult,
            run_id: str,
            runtime: RuntimeContext,
            query_understanding: QueryUnderstandingResult | None = None,
    ) -> DataPreparationResult:
        df = read_dataframe(data_path)
        operations = ["preserve_row_multiplicity"]

        fields = [field for field in request_analysis.selected_fields if field in df.columns]
        if fields:
            df = df[_unique(fields)].copy()
            operations.append(f"select_fields:{','.join(df.columns)}")

        for column in data_profile.likely_time_columns:
            if column in df.columns:
                df[column] = _parse_temporal(column, df[column])
                operations.append(f"to_datetime:{column}")

        for column in data_profile.likely_numeric_columns:
            if column in df.columns and df[column].isna().any():
                median = df[column].median()
                if pd.notna(median):
                    df[column] = df[column].fillna(median)
                    operations.append(f"fill_numeric_median:{column}")

        output_path = runtime.next_artifact_path("cleaned_data.csv", run_id=run_id)
        df.to_csv(output_path, index=False)
        return DataPreparationResult(output_path=output_path.as_posix(), operations=operations, row_count=len(df),
                                     col_count=len(df.columns))


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


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
