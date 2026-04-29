from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.domain.models import DataPreparationResult, DataProfile, QueryUnderstandingResult, RequestAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


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
        path = Path(data_path)
        df = self._read_frame(path)
        df = self._ensure_unique_columns(df)
        operations: list[str] = []

        selected_fields = [field for field in request_analysis.selected_fields if field in df.columns]
        if selected_fields:
            selected_fields = self._unique_preserve(selected_fields)
            df = df[selected_fields].copy()
            operations.append(f"select_fields:{','.join(selected_fields)}")

        should_drop_duplicates = self._should_drop_duplicates(df, data_profile, request_analysis, query_understanding)
        if should_drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            if len(df) != before:
                operations.append("drop_duplicates")
        else:
            operations.append("preserve_row_multiplicity")

        for column in [col for col in data_profile.likely_time_columns if col in df.columns]:
            try:
                df[column] = self._parse_temporal_column(column, df[column])
                operations.append(f"to_datetime:{column}")
            except Exception:
                pass

        for column in [col for col in data_profile.likely_numeric_columns if col in df.columns]:
            if df[column].isna().any():
                median = df[column].median()
                if pd.notna(median):
                    df[column] = df[column].fillna(median)
                    operations.append(f"fill_numeric_median:{column}")

        run_dir = runtime.ensure_run_dir(run_id)
        cleaned_path = run_dir / "cleaned_data.csv"
        df.to_csv(cleaned_path, index=False)
        return DataPreparationResult(
            output_path=cleaned_path.as_posix(),
            operations=operations,
            row_count=int(len(df)),
            col_count=int(len(df.columns)),
        )

    @staticmethod
    def _read_frame(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported data format: {suffix}")

    @staticmethod
    def _parse_temporal_column(column: str, series: pd.Series) -> pd.Series:
        if "year" in str(column).lower():
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().mean() >= 0.8:
                years = numeric.round().astype("Int64")

                def expand_year(value):
                    if pd.isna(value):
                        return pd.NA
                    value = int(value)
                    if 0 <= value <= 29:
                        return 2000 + value
                    if 30 <= value <= 99:
                        return 1900 + value
                    return value

                expanded = years.map(expand_year)
                return pd.to_datetime(expanded.astype("string") + "-01-01", errors="coerce")
        return pd.to_datetime(series, errors="coerce")

    @staticmethod
    def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
        if not df.columns.duplicated().any():
            return df
        counts: dict[str, int] = {}
        new_columns: list[str] = []
        for name in df.columns:
            counts[name] = counts.get(name, 0) + 1
            if counts[name] == 1:
                new_columns.append(name)
            else:
                new_columns.append(f"{name}__dup{counts[name] - 1}")
        copy = df.copy()
        copy.columns = new_columns
        return copy

    @staticmethod
    def _unique_preserve(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result

    @classmethod
    def _should_drop_duplicates(
            cls,
            df: pd.DataFrame,
            data_profile: DataProfile,
            request_analysis: RequestAnalysisResult,
            query_understanding: QueryUnderstandingResult | None,
    ) -> bool:
        """Preserve row multiplicity unless duplicate removal is explicitly requested."""
        task_text_parts: list[str] = []
        if query_understanding is not None:
            task_text_parts.extend([
                query_understanding.intent or "",
                " ".join(query_understanding.requested_operations or []),
                query_understanding.task_type or "",
                query_understanding.analysis_goal or "",
                query_understanding.user_goal or "",
            ])
        task_text_parts.extend(request_analysis.grounded_fields)
        task_text_parts.extend(request_analysis.normalization_hints)
        task_text = " ".join(task_text_parts).lower()
        explicit_markers = {
            "deduplicate",
            "deduplication",
            "remove duplicates",
            "drop duplicates",
            "unique rows",
            "without duplicates",
            "distinct rows",
        }
        return any(marker in task_text for marker in explicit_markers)
